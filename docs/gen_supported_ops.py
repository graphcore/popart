#!/usr/bin/python
# Copyright (c) 2018 Graphcore Ltd. All rights reserved.

import sys

if len(sys.argv) < 3:
    sys.exit(
        "gen_supported_ops.py <path of popart python install> <output file>")

print("Looking for popart module in " + sys.argv[1])
for p in sys.argv[1].split(':'):
    sys.path.append(p)
import popart

supported_ops = popart.getSupportedOperations(False)

ops = dict([[x, []] for x in set([x.domain for x in supported_ops])])
for op in supported_ops:
    ops[op.domain].append({"type": op.type, "version": op.version})

print("Writing supported ops to " + sys.argv[2])
with open(sys.argv[2], "w") as f:
    for domain in ops:
        print("Domain: " + domain, file=f)
        print('-' * (8 + len(domain)), file=f)
        print("", file=f)

        for op in ops[domain]:
            print(f"- {op['type']}-{op['version']}", file=f)

        print("", file=f)
