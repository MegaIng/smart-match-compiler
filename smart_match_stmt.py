from __future__ import annotations

import ast
from inspect import getsource, getsourcelines
from itertools import dropwhile
from types import FunctionType

from python_assembling import disassemble, BasicBlock, BasicBlockGraph

RAW_COLORS = "%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255"

HEX_COLORS = [e.removeprefix("%23")
              for e in RAW_COLORS.split('-')]
RGB_COLORS = [(int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
              for c in HEX_COLORS]


class MatchDecompiler(ast.NodeVisitor):
    def __init__(self, blocks: BasicBlockGraph):
        self.blocks = blocks
        self.inside_match = False
        self.branches = None
        self.else_block = None

    def visit_Match(self, node: ast.Match):
        assert not self.inside_match, "Can't deal with nested match statements"
        self.branches = []
        self.inside_match = True

        expr_start, expr_end = node.subject.lineno, node.subject.end_lineno
        expr_blocks = self.blocks.find_lines(range(expr_start, expr_end + 1))

        direct, jumps = self.blocks.outgoing(expr_blocks)
        assert len(direct) == 1, len(direct)
        self.else_block = direct[0].next
        assert all(j.jump_target[0] is self.else_block for j in jumps)
        for case in node.cases:
            self.visit(case)
        self.inside_match = False

    def visit_match_case(self, node: ast.match_case):
        assert self.inside_match, "match_case outside of match"
        assert node.pattern not in self.branches
        pattern_start, pattern_end = node.pattern.lineno, node.pattern.end_lineno
        start_body, end_body = node.body[0].lineno, node.body[-1].end_lineno
        pattern_blocks = self.blocks.find_lines(range(pattern_start, pattern_end + 1))
        body_blocks = self.blocks.find_lines(range(start_body, end_body + 1))

        self.branches.append((node.pattern, pattern_blocks, body_blocks))

    def get_colors(self) -> dict[BasicBlock, str]:
        out = {}
        for i, (p, pb, bb) in enumerate(self.branches):
            pc, bc = HEX_COLORS[(i * 2) % len(HEX_COLORS)], HEX_COLORS[(i * 2 + 1) % len(HEX_COLORS)]
            out.update(dict.fromkeys(pb, "#" + pc))
            out.update(dict.fromkeys(bb, "#" + bc))
        return out


def match2tree(func: FunctionType):
    lines, start_line = getsourcelines(func)
    assert start_line == func.__code__.co_firstlineno
    # inspect includes the annotations. We don't want them here
    lines = [(line if not line.strip().startswith('@') else '\n') for line in lines]
    # This makes sure we have the correct line numbers
    lines = ['\n'] * (start_line - 1) + lines
    source = ''.join(lines)
    print(func.__code__.co_firstlineno)
    tree = ast.parse(source, func.__code__.co_filename, mode='exec')
    blocks = disassemble(func.__code__)
    m2t = MatchDecompiler(blocks)
    m2t.visit(tree)
    blocks.to_pydot(m2t.get_colors()).write_png("graph.png")
    func.__code__ = blocks.assemble()
    return func
