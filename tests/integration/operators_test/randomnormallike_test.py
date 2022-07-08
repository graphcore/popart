# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import onnx
import json

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

# Parameterize over supported data types using a (numpy, onnx) datatype tuple
supported_dtypes = [
    (np.float32, onnx.helper.TensorProto.FLOAT),
    (np.float16, onnx.helper.TensorProto.FLOAT16),
]


# CPU test: Check that randomnormallike:
#  * copies type and shape from the input tensor
#  * is replaced by RandomNormal op in the IR
@pytest.mark.parametrize("dtypes", supported_dtypes)
def test_randomnormallike(dtypes):
    seed = 8
    data = np.zeros((1, 2, 10, 5), dtype=dtypes[0])

    builder = popart.Builder()
    T = builder.addInitializedInputTensor(data)
    out = builder.aiOnnx.randomnormallike([T])
    builder.addOutputTensor(out)

    with tu.create_test_device() as device:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
            deviceInfo=device,
        )

        session.prepareDevice()
        session.setRandomSeed(seed)
        session.weightsFromHost()

        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({}, anchors)
        session.run(stepio)

    # Check that the output has the correct shape and dtype
    assert anchors[out].shape == data.shape
    assert anchors[out].dtype == data.dtype

    # Check that the IR has RandomNormalLike replaced with RandomNormal
    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    graph = ir["maingraph"]

    like = [op for op in graph if op["type"] == "RandomNormalLike"]
    assert len(like) == 0, "Unexpected RandomNormalLike op in the IR."

    rn = [op for op in graph if op["type"] == "RandomNormal"]
    assert len(rn) == 1, "Expected one RandomNormal op in the IR."


# CPU test: check error when setting the optional seed
def test_randomnormallike_seederror(op_tester):
    def init_builder(builder):
        T = builder.addInputTensor(np.ones(10, dtype=np.float32))
        out = builder.aiOnnx.randomnormallike([T], seed=1.0)
        builder.addOutputTensor(out)
        return [out]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None)

    assert e_info.value.args[0].endswith(
        "Optional seed attribute is not supported. Use session::setRandomSeed instead."
    )


# CPU test: check error with unsupported data type
def test_randomnormallike_bad_dtype(op_tester):
    def init_builder(builder):
        T = builder.addInputTensor(np.ones(10, dtype=np.double))
        out = builder.aiOnnx.randomnormallike([T])
        builder.addOutputTensor(out)
        return [out]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None)

    assert "Unsupported data type requested" in e_info.value.args[0]


# IPU test: checks that the dtype attribute is used over the input tensor dtype.
# Also checks that the output has the expected statistical properties
@tu.requires_ipu
@pytest.mark.parametrize("dtypes", supported_dtypes)
def test_randomnormallike_stats(op_tester, dtypes):
    expected_mean = np.array(1.0, dtype=dtypes[0])
    expected_stddev = np.array(2.0, dtype=dtypes[0])
    numel = 10000
    N_1 = np.array(numel - 1, dtype=dtypes[0])
    data = np.zeros((numel, 1), dtype=dtypes[0])

    def init_builder(builder):
        T = builder.addInputTensor(data)
        rn = builder.aiOnnx.randomnormallike(
            [T], dtype=dtypes[1], mean=expected_mean, scale=expected_stddev
        )
        actual_mean = builder.aiOnnx.reducemean([rn], keepdims=False)
        builder.addOutputTensor(actual_mean)

        # var = stddev**2 = 1/(N-1) * sum( (x_i - mean)**2 )
        diff = builder.aiOnnx.sub([rn, actual_mean])
        sumsqdiff = builder.aiOnnx.reducesumsquare([diff], keepdims=False)
        denom = builder.addInputTensor(N_1)
        var = builder.aiOnnx.div([sumsqdiff, denom])
        actual_stddev = builder.aiOnnx.sqrt([var])
        builder.addOutputTensor(actual_stddev)

        return [actual_mean, actual_stddev]

    def reference(_):  # ref_data is an unused argument
        return [expected_mean, expected_stddev]

    op_tester.atol = 1e-01
    op_tester.run(init_builder, reference, seed=8)
