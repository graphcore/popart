# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

SHAPE = [1]
RANK = len(SHAPE)

# keep this in alphabetical order
# contains the different suffixes of ReduceOps for each opset
ONNX_OPS = [
    "l1", "l2", "logsum", "logsumexp", "max", "mean", "min", "prod", "sum",
    "sumsquare"
]
GRAPHCORE_OPS = ["median"]
INVALID_AXES = [-RANK - 1, RANK]


# builds a reduceOp from the specified opset and axes
# the axes provided are invalid, so the function throws an error when creating a session
def check_op_with_invalid_axes(opset, reduceOp, axis):
    with pytest.raises(popart.popart_exception) as e_info:
        builder = popart.Builder()
        x = builder.aiOnnx.constant(
            np.random.rand(*SHAPE).astype(np.float32), "FLOAT")

        # reducemedian returns 2 outputs in an array, so we convert the singleton outputs into arrays as well
        ys = getattr(getattr(builder, opset), reduceOp)([x], axes=[axis])
        if not isinstance(ys, list):
            ys = [ys]
        for y in ys:
            builder.addOutputTensor(y)
        anchors = {y: popart.AnchorReturnType("ALL") for y in ys}

        proto = builder.getModelProto()
        dataFlow = popart.DataFlow(1, anchors)
        device = popart.DeviceManager().createCpuDevice()

        session = popart.InferenceSession(proto, dataFlow,
                                          device)  # this should throw an error
    assert (e_info.value.args[0] == (
        "Axis {} is out of acceptable range [{}, {}]").format(
            axis, -RANK, RANK - 1))


@pytest.mark.parametrize("axis", INVALID_AXES)
@pytest.mark.parametrize("op", ONNX_OPS)
def test_reduce_invalid_axes_onnx_ops(axis, op):
    check_op_with_invalid_axes("aiOnnx", "reduce" + op, axis)


@pytest.mark.parametrize("axis", INVALID_AXES)
@pytest.mark.parametrize("op", GRAPHCORE_OPS)
def test_reduce_invalid_axes_graphcore_ops(axis, op):
    check_op_with_invalid_axes("aiGraphcore", "reduce" + op, axis)
