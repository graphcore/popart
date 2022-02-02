# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import json
# `import op_tester` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'operators_test'))
# pylint is disabled as op_tester is used as a fixture
from conftest import op_tester  # pylint: disable=unused-import


def test_issue(op_tester):
    lhs = [2, 384, 1024]
    rhs = [1024, 1024]

    def zeros(*args):
        return np.zeros(args, dtype=np.float32)

    print("matmul training test {} x {}".format(lhs, rhs))

    d1 = np.random.rand(*lhs).astype(np.float32)
    d2 = np.random.rand(*rhs).astype(np.float32)

    print("Result  {}".format(np.matmul(d1, d2).shape))

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        t1 = builder.aiOnnx.matmul([i1, i2])

        # loss can't handle scalar value produced by `matmul` of 2 1d tensors
        # so include an `add` operation, and put the output of matmul in anchors
        if np.matmul(d1, d2).shape == ():
            i3 = builder.addInputTensor(zeros(2))
            o = builder.aiOnnx.add([i3, t1])
            builder.addOutputTensor(o)
            return [
                o, t1,
                popart.reservedGradientPrefix() + o,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + i2
            ]
        else:
            builder.addOutputTensor(t1)
            return [
                t1,
                popart.reservedGradientPrefix() + t1,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + i2
            ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)

        r = torch.matmul(t1, t2)

        if r.shape == ():
            z1 = torch.tensor(zeros(2), requires_grad=True)
            out = z1 + r
            out__o = ref_data.getOutputTensorGrad(0)
            out.backward(torch.tensor(out__o))

            return [out, r, out__o, t1.grad, t2.grad]
        else:
            r__o = ref_data.getOutputTensorGrad(0)
            r.backward(torch.tensor(r__o))

            print("{} {} ".format(t1.grad, t2.grad))
            return [r, r__o, t1.grad, t2.grad]

    op_tester.patterns = popart.Patterns(popart.PatternsLevel.Default)
    op_tester.options.enableOutlining = False
    session = op_tester.run(init_builder, reference, 'train')

    # Check the ir
    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    # There should be no ReduceSum in the ir
    assert 'ReduceSum' not in [op['type'] for op in ir['maingraph']]
    # Get the 3 matmuls, 1 in the forward pass and 2 generated in the backward pass
    matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
    assert len(matmuls) == 3
    # Check that the first two dimensions of the matmul (2 and 384) were combined (768).
    assert matmuls[0]['inputs'][0]['shape'] == '[1 768 1024]'
