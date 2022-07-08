# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
import torch.nn as nn

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu


@pytest.mark.parametrize(
    "npSrcType,npDstType,builderDstType,reducePrecision",
    [
        (np.int32, np.float32, "FLOAT", False),
        (np.float32, np.int32, "INT32", False),
        (np.int16, np.float32, "FLOAT", False),
        (np.float32, np.int16, "INT16", False),
        (np.int16, np.float16, "FLOAT16", False),
        (np.float16, np.int16, "INT16", False),
        (np.uint16, np.float16, "FLOAT16", False),
        (np.float16, np.uint16, "UINT16", False),
        (np.int8, np.float16, "FLOAT16", False),
        (np.float16, np.int8, "INT8", False),
        (np.uint8, np.float16, "FLOAT16", False),
        (np.float16, np.uint8, "UINT8", False),
        (np.float32, np.float16, "FLOAT16", True),
        (np.float16, np.float32, "FLOAT", False),
        (np.int32, np.int32, "INT32", False),
    ],
)
def test_cast(op_tester, npSrcType, npDstType, builderDstType, reducePrecision):
    d1 = np.random.uniform(0, 20, 5).astype(npSrcType)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.cast([i1], builderDstType)

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        return [d1.astype(npDstType)]

    if npSrcType in (np.int8, np.uint8):
        op_tester.options.opxModifyChecking = False

    if reducePrecision:  # Some downcasts change values enough to fail for 1e-8
        op_tester.atol = 1e-2

    op_tester.run(init_builder, reference, "infer")


@pytest.mark.parametrize(
    "npSrcType,torchDstType,builderDstType",
    [
        (np.float32, torch.float32, "FLOAT"),
        (np.float16, torch.float32, "FLOAT"),
        (np.float32, torch.float16, "FLOAT16"),
        (np.float16, torch.float16, "FLOAT16"),
    ],
)
def test_cast_grad(op_tester, npSrcType, torchDstType, builderDstType):
    d1 = np.random.uniform(0, 10, 10).astype(npSrcType)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.cast([i1], builderDstType)
        # Add an op that produces a gradient so we can test CastGrad properly
        o = builder.aiOnnx.reducesum([c])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        c = torch.tensor(d1, dtype=torchDstType, requires_grad=True)
        out = torch.sum(c).reshape((1,))
        d_o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d_o))
        d_i1 = c.grad.numpy().astype(npSrcType)
        return [out, d_i1, d_o]

    op_tester.setPatterns(
        ["PreUniRepl", "PostNRepl", "SqrtGradOp"], enableRuntimeAsserts=False
    )
    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize(
    "npSrcType", [np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32]
)
@pytest.mark.parametrize("builderDstType", ["FLOAT", "FLOAT16"])
def test_cast_no_grad(npSrcType, builderDstType):
    """Check that CastOp, doesn't return gradient Op when casted-from type is
    not float/half.
    """
    np.random.seed(0)
    # Is randomly generated data ok here? Also, the tested range is [0, 10], so
    # no negative inputs are tested.
    inputData = np.random.uniform(0, 10, 10).astype(npSrcType)

    builder = popart.Builder()
    input_ = builder.addInputTensor(popart.TensorInfo(inputData))
    output_ = builder.aiOnnx.cast([input_], builderDstType)
    builder.addOutputTensor(output_)
    lossId = builder.aiGraphcore.identityloss([output_])
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(
        1,
        {
            output_: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + input_: popart.AnchorReturnType("All"),
        },
    )

    with tu.create_test_device() as device:

        patterns = popart.Patterns(
            ["PreUniRepl", "PostNRepl", "SqrtGradOp"]
        ).enableRuntimeAsserts(False)
        options = popart.SessionOptions()
        options.enableStochasticRounding = False

        with pytest.raises(popart.popart_exception) as e_info:
            popart.TrainingSession(
                fnModel=proto,
                loss=lossId,
                dataFlow=dataFlow,
                deviceInfo=device,
                optimizer=popart.ConstSGD(0.01),
                patterns=patterns,
                userOptions=options,
            )

        assert e_info.value.args[0].startswith(
            f"Anchor tensor `{popart.reservedGradientPrefix() + input_}' not in Ir Tensors."
        )


def test_cast_no_grad_branch(op_tester):
    r"""Check if CastOp from INT32 stops the gradient propagation in a graph.

    The following example is used:

        FWD:                  BWD:

        Input                Input'
        /   \                 ^
       /     \               /
     Pow     Pow           Pow'
      |       |             ^
      |       |             |
      |    CastInt          |
      |       |             |
      |       |             |
      |   CastFloat         |
       \     /               \-
        -Add-                 Add'
          |                    ^
          |                    |
       Softmax              Softmax'
          |                    ^
          |                    |
        Output               Output'

    The right branch is not used in the calculation of the gradient.
    """
    data = np.random.rand(4, 4).astype(np.float32)
    w_data = np.random.rand(4, 4).astype(np.float32)
    two = np.array([2]).astype(np.float32)

    def init_builder(builder):
        i = builder.addInputTensor(data)
        two_i = builder.aiOnnx.constant(two, "two")
        w = builder.addInitializedInputTensor(w_data)

        a1 = builder.aiOnnx.pow([i, two_i])
        b1 = builder.aiOnnx.pow([i, two_i])
        b1 = builder.aiOnnx.cast([b1], "INT32")
        b1 = builder.aiOnnx.cast([b1], "FLOAT")
        o = builder.aiOnnx.sum([a1, b1, w])
        o = builder.aiOnnx.softmax([o], axis=1)

        builder.addOutputTensor(o)

        return [
            o,
            popart.reservedGradientPrefix() + i,
            popart.reservedGradientPrefix() + w,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        x = torch.tensor(data, requires_grad=True)
        w_t = torch.tensor(w_data, requires_grad=True)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.weight_param = w_t

            def forward(self, x):
                pow1 = x ** 2
                with torch.no_grad():
                    pow2 = x ** 2
                    pow2 = pow2.to(torch.int32)
                    pow2 = pow2.to(torch.float32)
                out = pow1 + pow2 + self.weight_param
                out = torch.softmax(out, dim=1)
                return out

        net = Net()
        out = net(x)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, x.grad, net.weight_param.grad, None]

    op_tester.setPatterns(["PowArg0GradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")
