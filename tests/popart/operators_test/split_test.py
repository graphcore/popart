from op_tester import op_tester
import numpy as np
import popart
import torch


def test_split_basic(op_tester):
    data = np.random.rand(4, 4).astype(np.float32)

    def init_builder(builder):
        i = builder.addInputTensor(data)

        a, b = builder.aiOnnx.split([i], 2)
        o = builder.aiOnnx.sum([a, b])

        builder.addOutputTensor(o)

        return [o]

    def reference(ref_data):
        xs = np.split(data, 2)
        out = xs[0] + xs[1]
        return [out]

    op_tester.run(init_builder, reference)


def test_split_custom_lengths(op_tester):
    data = np.random.rand(6).astype(np.float32)

    def init_builder(builder):
        i = builder.addInputTensor(data)

        a, b, c = builder.aiOnnx.split([i], 3, 0, (1, 2, 3))

        builder.addOutputTensor(a)
        builder.addOutputTensor(b)
        builder.addOutputTensor(c)

        return [a, b, c]

    def reference(ref_data):
        xs = np.split(data, (1, 3))

        return xs

    op_tester.run(init_builder, reference)


def test_split_grad(op_tester):
    data = np.random.rand(4, 4).astype(np.float32)

    def init_builder(builder):
        i = builder.addInputTensor(data)

        a, b = builder.aiOnnx.split([i], 2)
        o = builder.aiOnnx.sum([a, b])

        builder.addOutputTensor(o)

        return [
            o,
            popart.reservedGradientPrefix() + i,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        x = torch.tensor(data, requires_grad=True)
        xs = torch.split(x, 2)
        out = xs[0] + xs[1]

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, x.grad, None]

    op_tester.passes = ['SplitGradOpToConcat']
    op_tester.run(init_builder, reference, 'train')
