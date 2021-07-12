from __future__ import annotations

import difflib
from dis import dis
from io import StringIO
from types import CodeType


from smart_match_stmt import match2tree

def view_dis(f_or_none=None, /, **kwargs):
    def wrapper(f):
        dis(f, **kwargs)
        return f
    if f_or_none is None:
        return wrapper
    else:
        return wrapper(f)


@view_dis(file=open("out.dis", "w"))
@match2tree
@view_dis(file=open("in.dis","w"))
def match_command(command):
    match (command.split() if command else "default"):
        case ["north"] | ["go", "north"]:
            print("north")
        case ["get", obj] | ["pick", "up", obj] | ["pick", obj, "up"]:
            print("pick up", obj)
    return None


match_command("go north")
match_command("pick up key")
match_command("get branch")

# f = match_command
# 
# dis(f, file=open("original.dis", "w"))
# 
# blocks = list(disassemble(f.__code__))
# # blocks[0].code_info.varnames[0] = "dadsaf"
# code = assemble(blocks)
# 
# g = to_pydot(blocks)
# g.write_png("graph.png")
# 
# print(f.__code__.co_code)
# print(code.co_code)
# print(code.co_code == f.__code__.co_code)
# print(list(f.__code__.co_lines()))
# print(list(code.co_lines()))
# 
# dis(code, file=open("recreated.dis", "w"))