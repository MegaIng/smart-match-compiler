from __future__ import annotations

import dis
import opcode as opc
from dataclasses import dataclass
from opcode import EXTENDED_ARG, HAVE_ARGUMENT
from types import CodeType
from typing import Any, Optional


class Instruction:
    def __init__(self, op: int | str, argument: Any = None):
        if isinstance(op, int):
            self.opcode = op
        else:
            self.opcode = opc.opmap[op]
        self.argument = argument

    @property
    def opname(self):
        return opc.opname[self.opcode]

    @opname.setter
    def opname(self, value):
        self.opcode = opc.opmap[value]

    def generate_with_extended(self, arg: Optional[int]) -> list[dis.Instruction]:
        if arg is None:
            return []
        out = []
        while arg > 0:
            out.append(arg & 0xFF)
            arg >>= 8
        out.reverse()
        if not out:
            out.append(0)
        return [
            *(dis.Instruction('EXTENDED_ARG', EXTENDED_ARG, val, val, repr(val), None, None, None)
              for val in out[:-1]),
            dis.Instruction(self.opname, self.opcode, out[-1], self.argument, repr(self.argument), None, None, None)
        ]


@dataclass(eq=False)
class CodeInfo:
    argcount: int
    posonlyargcount: int
    kwonlyargcount: int
    nlocals: int
    stacksize: int
    flags: int
    firstlineno: int
    consts: list[Any]
    names: list[str]
    varnames: list[str]
    filename: str
    name: str
    freevars: list[str]
    cellvars: list[str]

    def __post_init__(self):
        all_names = set(self.varnames)
        for n in (self.freevars, self.cellvars):
            if not all_names.isdisjoint(n):
                raise ValueError(f"Name {n!r} in different groups")
            all_names.update(n)

    @classmethod
    def from_code(cls, code: CodeType) -> CodeInfo:
        return cls(
            argcount=code.co_argcount,
            posonlyargcount=code.co_posonlyargcount,
            kwonlyargcount=code.co_kwonlyargcount,
            nlocals=code.co_nlocals,
            stacksize=code.co_stacksize,
            flags=code.co_flags,
            firstlineno=code.co_firstlineno,
            consts=list(code.co_consts),
            names=list(code.co_names),
            varnames=list(code.co_varnames),
            filename=code.co_filename,
            name=code.co_name,
            freevars=list(code.co_freevars),
            cellvars=list(code.co_cellvars),
        )

    def _to_bytes(self, instructions: list[dis.Instruction]) -> tuple[bytes, bytes]:
        code = bytearray()
        linetab = bytearray()
        full_arg = 0
        line_ranges = []
        current_range_start = 0
        current_line_number = self.firstlineno
        current_line_offset = 0
        offset = 0
        for i, ins in enumerate(instructions):
            offset = 2 * i
            assert len(code) == offset

            # Full argument value
            if ins.opcode == EXTENDED_ARG:
                if full_arg is None:
                    full_arg = 0
                full_arg = full_arg << 8 | (ins.arg & 0xFF)
            elif ins.opcode > HAVE_ARGUMENT:
                full_arg = ins.arg
            else:
                full_arg = None

            # Bytecode
            assert 0 <= ins.opcode <= 0xFF
            code.append(ins.opcode)
            code.append(ins.arg & 0xFF if ins.arg is not None else 0)

            # linetab
            if ins.starts_line:
                line_ranges.append((offset - current_range_start, current_line_offset))
                current_range_start = offset
                current_line_offset = ins.starts_line - current_line_number
                current_line_number = ins.starts_line
        line_ranges.append((offset + 2 - current_range_start, current_line_offset))

        for ins_offset, line_offset in line_ranges:
            d = 1 if line_offset > 0 else -1
            line_offset = abs(line_offset)
            while line_offset > 127:
                linetab.extend((0, 127 * d % 256))
                line_offset -= 127
            while ins_offset > 255:
                linetab.extend((255, 0))
                ins_offset -= 127
            linetab.extend((ins_offset, (line_offset * d) % 256))
        return bytes(code), bytes(linetab)

    def built(self, instructions: list[dis.Instruction], stacksize: int = None):
        if stacksize is None:
            stacksize = self.stacksize
        code, linetab = self._to_bytes(instructions)
        return CodeType(
            self.argcount,
            self.posonlyargcount,
            self.kwonlyargcount,
            self.nlocals,
            stacksize,
            self.flags,
            code,
            tuple(self.consts),
            tuple(self.names),
            tuple(self.varnames),
            self.filename,
            self.name,
            self.firstlineno,
            linetab,
            tuple(self.freevars),
            tuple(self.cellvars)
        )

    def get_const_id(self, value: Any) -> int:
        if value not in self.consts:
            self.consts.append(value)
        return self.consts.index(value)

    def get_name_id(self, name: str) -> int:
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name)

    def get_local_id(self, name: str) -> int:
        if name not in self.varnames:
            self.varnames.append(name)
        return self.varnames.index(name)

    def get_free_id(self, name: str) -> int:
        if name in self.freevars:
            return self.freevars.index(name)
        elif name in self.cellvars:
            return len(self.freevars) + self.cellvars.index(name)
        else:
            raise ValueError(name)

    def transpile(self, ins: Instruction) -> list[dis.Instruction]:
        if ins.opcode in opc.hasconst:
            arg = self.get_const_id(ins.argument)
        elif ins.opcode in opc.hasname:
            arg = self.get_name_id(ins.argument)
        elif ins.opcode in opc.hasjrel:
            raise ValueError("Jumps need to be handle at a higher level")
        elif ins.opcode in opc.hasjabs:
            raise ValueError("Jumps need to be handle at a higher level")
        elif ins.opcode in opc.haslocal:
            arg = self.get_local_id(ins.argument)
        elif ins.opcode in opc.hascompare:
            arg = opc.cmp_op.index(ins.argument)
        elif ins.opcode in opc.hasfree:
            arg = self.get_free_id(ins.argument)
        else:
            arg = ins.argument
        if ins.opcode < HAVE_ARGUMENT and arg is not None:
            raise ValueError(f"Opcode {ins.opname} can't have an argument")
        return ins.generate_with_extended(arg)

