from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from dis import Instruction, opmap, opname, get_instructions
from opcode import EXTENDED_ARG, hasjabs, hasjrel, HAVE_ARGUMENT, hasconst, hasname, hasfree, hascompare, cmp_op, haslocal, stack_effect
from types import CodeType
from typing import Iterable, Any, Optional, Collection, Container


@dataclass(eq=False)
class ExtendedInstruction:
    code_info: CodeInfo
    ins: Instruction
    arg_values: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.ins = self.ins._replace(
            argval=None, argrepr=None, offset=None, is_jump_target=None
        )

    @property
    def opname(self):
        return self.ins.opname

    @opname.setter
    def opname(self, value):
        opcode = opmap[value]
        self.ins = self.ins._replace(opname=value, opcode=opcode)

    @property
    def opcode(self):
        return self.ins.opcode

    @opcode.setter
    def opcode(self, value):
        opn = opname[value]
        self.ins = self.ins._replace(opname=opn, opcode=value)

    @property
    def arg(self) -> int:
        return self.ins.arg

    @arg.setter
    def arg(self, value: int):
        self.ins = self.ins._replace(arg=value)

    @property
    def raw_instructions(self) -> list[Instruction]:
        if self.ins.arg is None:
            return [self.ins]
        out = []
        v = self.ins.arg
        while v > 0:
            out.append(v & 0xFF)
            v >>= 8
        out.reverse()
        if not out:
            out.append(0)
        return [
            *(Instruction('EXTENDED_ARG', EXTENDED_ARG, val, val, None, None, None, None)
              for val in out[:-1]),
            self.ins._replace(arg=out[-1])
        ]

    @property
    def argval(self):
        if self.opcode < HAVE_ARGUMENT:
            return None
        elif self.opcode in hasjrel or self.opcode in hasjabs:
            return self.arg
        elif self.opcode in hasconst:
            return self.code_info.consts[self.arg]
        elif self.opcode in hasname:
            return self.code_info.names[self.arg]
        elif self.opcode in hasfree:
            return self.code_info.freevars[self.arg]
        elif self.opcode in hascompare:
            return cmp_op[self.arg]
        elif self.opcode in haslocal:
            return self.code_info.varnames[self.arg]
        else:
            return self.arg

    def summary(self):
        if self.arg is None:
            s = self.opname
        else:
            s = f"{self.opname} {self.arg} ({self.argval!r})"
        # if self.ins.starts_line is not None:
        #     s = f"({self.ins.starts_line:2}) {s}"
        # else:
        #     s = f"       {s}"
        return s


def _to_bytes(instructions: list[Instruction], firstline: int) -> tuple[bytes, bytes]:
    code = bytearray()
    linetab = bytearray()
    full_arg = 0
    line_ranges = []
    current_range_start = 0
    current_line_number = firstline
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

    def built(self, instructions: list[Instruction], stacksize: int = None):
        if stacksize is None:
            stacksize = self.stacksize
        code, linetab = _to_bytes(instructions, self.firstlineno)
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


@dataclass(eq=False)
class BasicBlock:
    code_info: CodeInfo
    line: int = None
    instructions: list[ExtendedInstruction] = field(default_factory=list)
    jump_instruction: ExtendedInstruction = None

    jump_target: tuple[BasicBlock, int] = None
    next: BasicBlock = None

    def raw_instructions(self, start_offset: int, target_offset: int) -> Iterable[Instruction]:
        current_offset = start_offset
        for ins in self.instructions:
            r = ins.raw_instructions
            current_offset += len(r) * 2
            yield from r
        if self.jump_instruction is None:
            assert self.jump_target is None
            return
        if self.jump_instruction.opcode in hasjabs:
            self.jump_instruction.arg = target_offset // 2 + self.jump_target[1]
        else:
            assert self.jump_instruction.opcode in hasjrel, self.jump_instruction
            self.jump_instruction.arg = (target_offset - current_offset) // 2 + self.jump_target[1] - 1
        yield from self.jump_instruction.raw_instructions

    def summary(self):
        out = []
        for eins in self.instructions:
            out.append(eins.summary())
        if self.jump_instruction is not None:
            out.append(self.jump_instruction.summary())
        # w = max(map(len, out))
        return f"<line {self.line}>\n" + "\n".join(out)


@dataclass(eq=False)
class BlockGraph:
    code_info: CodeInfo
    entry_block: BasicBlock
    blocks: list[BasicBlock]
    
    def find_lines(self, lines: Container[int]) -> list[BasicBlock]:
        return [b for b in self.blocks if b.line in lines]

    def pred(self, block: BasicBlock) -> tuple[Optional[BasicBlock], list[BasicBlock, int]]:
        direct = None
        jumps = []
        for b in self.blocks:
            if b.next is block:
                assert direct is None
                direct = b
            if b.jump_target is not None and b.jump_target[0] is block:
                jumps.append(b)
        return direct, jumps
    
    def outgoing(self, blocks: list[BasicBlock]) -> tuple[list[BasicBlock], list[BasicBlock]]:
        direct = []
        jumps = []
        for b in blocks:
            if b.next is not None and b.next not in blocks:
                direct.append(b)
            if b.jump_target is not None and b.jump_target[0] not in blocks:
                jumps.append(b)
        return direct, jumps

    def correct_order(self):
        sections_by_first = {}
        sections_by_next = {}
        for b in self.blocks:
            section_next = sections_by_next.pop(b, None)
            section_first = sections_by_first.pop(b.next, None)
            if section_first is None is section_next:
                section = [b]
            elif section_first is None:
                section = section_next
                section.append(b)
            elif section_next is None:
                section = section_first
                section.insert(0, b)
            else:
                sections_by_next.pop(section_first[-1], None)
                sections_by_first.pop(section_next[0], None)
                section = [*section_next, b, *section_first]
            if section[-1].next is not None:
                sections_by_next[section[-1].next] = section
            sections_by_first[section[0]] = section
        # print(sections_by_next, sections_by_first)
        assert not sections_by_next, "Some sections did not get completed"

        assert self.entry_block in sections_by_first
        new_order = sections_by_first[self.entry_block]

        # TODO: Here we could (should?) do some fancy ordering.
        for b, s in sections_by_first.items():
            if b is not self.entry_block:
                new_order.extend(s)

        assert len(new_order) == len(set(new_order))
        assert set(new_order) == set(self.blocks)
        self.blocks = new_order

    def to_pydot(self):
        import pydot
        graph = pydot.Dot(graph_type='digraph', labelloc="l")
        nodes = {}
        for block in self.blocks:
            graph.add_node(pydot.Node(hex(id(block)), label=block.summary().replace("\n", "\\l") + "\\l"))
            if block.next is not None:
                graph.add_edge(pydot.Edge(hex(id(block)), hex(id(block.next)), label="next"))
            if block.jump_target is not None:
                jn, i = block.jump_target
                graph.add_edge(pydot.Edge(hex(id(block)), hex(id(jn)), label=f"jump to {i}"))
        return graph

    def assemble(self) -> CodeType:
        self.correct_order()
        # Sanity check
        last = self.blocks[0]
        for b in self.blocks[1:]:
            assert b.code_info is self.code_info
            assert last.next in (None, b)
            last = b
        assert last.next is None
        assert self.entry_block is self.blocks[0]

        offset_map = {}

        change = 5
        instructions = []
        while change:
            change -= 1
            instructions = []
            for b in self.blocks:
                current_offset = len(instructions) * 2
                if b not in offset_map or (offset_map[b] != current_offset):
                    change = max(change, 1)
                    offset_map[b] = current_offset
                if b.jump_target is not None:
                    target_offset = offset_map.get(b.jump_target[0], current_offset)
                else:
                    target_offset = None
                raw = b.raw_instructions(current_offset, target_offset)
                instructions.extend(raw)
        return self.code_info.built(instructions)


exit_function = ['RERAISE', 'RETURN_VALUE', 'RAISE_VARARGS']
exit_function = [opmap[n] for n in exit_function]
unconditional = ['JUMP_FORWARD', 'JUMP_ABSOLUTE']
unconditional = [opmap[n] for n in unconditional]


def disassemble(code: CodeType) -> BlockGraph:
    code_info = CodeInfo.from_code(code)
    blocks = {}
    block_start = 0
    targets = {}
    missing_targets = defaultdict[int, list[BasicBlock]](list)
    current_line = code.co_firstlineno

    prev = None
    current = None

    arg_values = []
    _argvalue = 0
    for ins in get_instructions(code):
        if ins.starts_line is not None:
            prev = current
            current = None
            current_line = ins.starts_line
        if current is None:
            current = blocks[ins.offset] = BasicBlock(code_info, current_line)
            if prev is not None:
                prev.next = current
                prev = None
            block_start = ins.offset
        if ins.opcode == EXTENDED_ARG:
            arg_values.append(ins.arg)
            _argvalue = (_argvalue << 8) | ins.arg
        else:
            if ins.arg is not None:
                assert (_argvalue | (ins.arg & 0xFF)) == ins.arg
            if ins.is_jump_target:
                targets[ins.offset] = current, (ins.offset - block_start) // 2
                for b in missing_targets[ins.offset]:
                    b.jump_target = targets[ins.offset]
                del missing_targets[ins.offset]
            eins = ExtendedInstruction(code_info, ins, arg_values)
            if ins.opcode in hasjrel or ins.opcode in hasjabs:
                current.jump_instruction = eins
                if ins.argval not in targets:
                    missing_targets[ins.argval].append(current)
                else:
                    current.jump_target = targets[ins.argval]
                if ins.opcode not in unconditional:
                    prev = current
                current = None
            elif ins.opcode in exit_function:
                current.instructions.append(eins)
                current = None
            else:
                current.instructions.append(eins)

    return BlockGraph(code_info, blocks[0], list(blocks.values()))