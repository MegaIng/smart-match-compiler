from __future__ import annotations

import ast
from inspect import getsource, getsourcelines
from itertools import dropwhile
from types import FunctionType

from python_assembling import disassemble, BasicBlock, BlockGraph


class Match2Tree(ast.NodeVisitor):
    def __init__(self, blocks: BlockGraph):
        self.blocks = blocks
        self.inside_match = False
        self.branches = None

    def visit_Match(self, node: ast.Match):
        assert not self.inside_match, "Can't deal with nested match statements"
        self.branches = []
        self.inside_match = True

        expr_start, expr_end = node.subject.lineno, node.subject.end_lineno
        expr_blocks = self.blocks.find_lines(range(expr_start, expr_end + 1))
        print(expr_start, expr_end, len(expr_blocks))
        direct, jumps = self.blocks.outgoing(expr_blocks)
        assert len(direct) == 1, len(direct)
        entry = direct[0].next
        assert all(j.jump_target[0] is entry for j in jumps)
        for case in node.cases:
            self.visit(case)
        self.inside_match = False

    def visit_match_case(self, node: ast.match_case):
        assert self.inside_match, "match_case outside of match"
        assert node.pattern not in self.branches
        self.branches.append(node)
        pattern_start, pattern_end = node.pattern.lineno, node.pattern.end_lineno
        start_body, end_body = node.body[0].lineno, node.body[-1].end_lineno
        pattern_blocks = self.blocks.find_lines(range(pattern_start, pattern_end + 1))
        body_blocks = self.blocks.find_lines(range(start_body, end_body + 1))
        print(pattern_start, pattern_end, start_body, end_body)


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
    blocks.to_pydot().write_png("graph.png")
    Match2Tree(blocks).visit(tree)
    func.__code__ = blocks.assemble()
    return func
