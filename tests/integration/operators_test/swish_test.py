# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import torch



@pytest.mark.skipif(not hasattr(torch.nn, 'SiLU'),
                    reason='Remove after T38031 is resolved')
def test_swish(op_tester):
    data = np.linspace(
        np.finfo(np.float32).min,
        np.finfo(np.float32).max,
        num=240,
        dtype=np.float32,
    ).reshape((5, 3, 4, 4))

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        out = builder.aiGraphcore.swish(
            [tensor],
            debugContext='test_swish',
        )
        builder.addOutputTensor(out)
        return [out]

    def reference(ref_data):
        # Note that torch silu on CPU is not implemented for half.
        swish = torch.nn.SiLU()
        return [swish(torch.tensor(data))]

    op_tester.run(init_builder, reference, 'infer')


@pytest.mark.skipif(not hasattr(torch.nn, 'SiLU'),
                    reason='Remove after T38031 is resolved')
def test_swish_grad(op_tester):
    data = np.linspace(
        np.finfo(np.float32).min,
        np.finfo(np.float32).max,
        num=240,
        dtype=np.float32,
    ).reshape((5, 3, 4, 4))

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        out = builder.aiGraphcore.swish(
            [tensor],
            debugContext='test_swish_grad',
        )
        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + out,
            popart.reservedGradientPrefix() + tensor,
        ]

    def reference(ref_data):
        # Note that torch silu on CPU is not implemented for half.
        swish = torch.nn.SiLU()
        tensor = torch.tensor(data)
        tensor.requires_grad = True
        out = swish(tensor)
        out.backward(torch.tensor(ref_data.getOutputTensorGrad(0)))
        return [out, out.grad, tensor.grad]

    op_tester.run(init_builder, reference, 'train')


def test_swish_shape_infer():
    data = np.random.randn(5, 3, 4, 4).astype(np.float32)
    builder = popart.Builder()
    tensor = builder.addInputTensor(popart.TensorInfo(data))
    out = builder.aiGraphcore.swish(
        [tensor],
        debugContext='test_swish_grad',
    )
    builder.addOutputTensor(out)
    assert builder.getTensorShape(out) == [5, 3, 4, 4]
    assert builder.getTensorDtypeString(out) == "float32"
