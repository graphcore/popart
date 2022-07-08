#!/usr/bin/python
# Copyright (c) 2018 Graphcore Ltd. All rights reserved.

import sys

if len(sys.argv) < 2:
    sys.exit("gen_popart_supported_ops.py <output file>")

import popart

supported_ops = popart.getSupportedOperations(False)

ops = dict([[x, []] for x in set([x.domain for x in supported_ops])])
for op in supported_ops:
    ops[op.domain].append({"type": op.type, "version": op.version})

print("Writing supported ops to " + sys.argv[1])
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for domain in ops:
        print("Domain: " + domain, file=f)
        print("-" * (8 + len(domain)), file=f)
        print("", file=f)

        for op in ops[domain]:
            print(f"- {op['type']}-{op['version']}", file=f)

        print("", file=f)
