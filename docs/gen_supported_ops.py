#!/usr/bin/python

import poponnx

supported_ops = poponnx.getSupportedOperations(False)

print("Poponnx supported operators")
print("===========================")
print("")

ops = dict([[x, []] for x in set([x[1] for x in supported_ops])])
for op in supported_ops:
    ops[op[1]].append(op[0])

for domain in ops:
    print("Domain: " + domain)
    print("---------------")

    for op in ops[domain]:
        print("- " + op)

    print("")
