# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch


def _test_matmul_broadcasting_base(op_tester, shapes):
    def zeros(*args):
        return np.zeros(args, dtype=np.float32)

    # Test for inference
    for lhs, rhs in shapes:

        print("matmul inference test {} x {} = ".format(lhs, rhs))

        d1 = np.random.rand(*lhs).astype(np.float32)
        d2 = np.random.rand(*rhs).astype(np.float32)
        print(" result  {}".format(np.matmul(d1, d2).shape))

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
                return [o, t1]
            else:
                builder.addOutputTensor(t1)
                return [t1]

        def reference(ref_data):
            t1 = np.matmul(d1, d2)
            if t1.shape == ():
                out = zeros(2) + t1
                return [out, np.array(t1)]
            else:
                return [t1]

        op_tester.run(init_builder, reference, 'infer')

    # Verify with torch
    print("")
    for lhs, rhs in shapes:
        print("matmul torch training test {} x {}".format(lhs, rhs))
        d1 = np.random.rand(*lhs).astype(np.float32)
        d2 = np.random.rand(*rhs).astype(np.float32)

        d3 = np.matmul(d1, d2)

        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        r = torch.matmul(t1, t2)

        loss = (t3 - r).pow(2).sum()

        loss.backward()

        print(" - output {}".format(t3.shape))
        print(" - d_Lhs {} d_Rhs {} ".format(t1.grad.shape, t2.grad.shape))

    # Test for training
    for lhs, rhs in shapes:

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

        # Test with the MatMulXXGradOp to MatMulOp pass
        op_tester.patterns = popart.Patterns(
            popart.PatternsLevel.Minimal).enablePattern(
                "MatMulLhsGradOp", True).enablePattern("MatMulRhsGradOp", True)

        op_tester.run(init_builder, reference, 'train')
