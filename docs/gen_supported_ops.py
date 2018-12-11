#!/usr/bin/python

import sys

if len(sys.argv) < 3:
    sys.exit(
        "gen_supported_ops.py <path of poponnx python install> <output file>")

print("Looking for poponnx module in " + sys.argv[1])
print("Writing supported ops to " + sys.argv[2])

sys.path.append(sys.argv[1])
import poponnx

supported_ops = poponnx.getSupportedOperations(False)

ops = dict([[x, []] for x in set([x[1] for x in supported_ops])])
for op in supported_ops:
    ops[op[1]].append(op[0])

with open(sys.argv[2], "w") as f:
    for domain in ops:
        print("Domain: " + domain, file=f)
        print("~~~~~~~~~~~~~~\n", file=f)

        for op in ops[domain]:
            print("- " + op, file=f)

        print("", file=f)
