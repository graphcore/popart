# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import onnx
import json
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

# Parameterize over supported data types using a (numpy, onnx) datatype tuple
supported_dtypes = [(np.float32, onnx.helper.TensorProto.FLOAT),
                    (np.float16, onnx.helper.TensorProto.FLOAT16)]


# CPU test: Check that randomuniformlike:
#  * copies type and shape from the input tensor
#  * is replaced by RandomUniform op in the IR
@pytest.mark.parametrize("dtypes", supported_dtypes)
def test_randomuniformlike(dtypes):
    seed = 8
    data = np.zeros((1, 2, 10, 5), dtype=dtypes[0])

    builder = popart.Builder()
    T = builder.addInitializedInputTensor(data)
    out = builder.aiOnnx.randomuniformlike([T])
    builder.addOutputTensor(out)

    session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
        deviceInfo=tu.create_test_device())

    session.prepareDevice()
    session.setRandomSeed(seed)
    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({}, anchors)
    session.run(stepio)

    # Check that the output has the correct shape and dtype
    assert anchors[out].shape == data.shape
    assert anchors[out].dtype == data.dtype

    # Check that the IR has RandomUniformLike replaced with RandomUniform
    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    graph = ir['maingraph']

    like = [op for op in graph if op['type'] == 'RandomUniformLike']
    assert len(like) == 0, "Unexpected RandomUniformLike op in the IR."

    rn = [op for op in graph if op['type'] == 'RandomUniform']
    assert len(rn) == 1, "Expected one RandomUniform op in the IR."


# CPU test: check error when setting the optional seed
def test_randomuniformlike_seederror(op_tester):
    def init_builder(builder):
        T = builder.addInputTensor(np.ones(10, dtype=np.float32))
        out = builder.aiOnnx.randomuniformlike([T], seed=1.0)
        builder.addOutputTensor(out)
        return [out]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None)

    assert (e_info.value.args[0].endswith(
        "Optional seed attribute is not supported. Use session::setRandomSeed instead."
    ))


# CPU test: check error with unsupported data type
def test_randomuniformlike_bad_dtype(op_tester):
    def init_builder(builder):
        T = builder.addInputTensor(np.ones(10, dtype=np.double))
        out = builder.aiOnnx.randomuniformlike([T])
        builder.addOutputTensor(out)
        return [out]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None)

    assert ("Unsupported data type requested" in e_info.value.args[0])


# IPU test: checks that the dtype attribute is used over the input tensor dtype.
# Also checks that the output has the expected statistical properties
@tu.requires_ipu
@pytest.mark.parametrize("dtypes", supported_dtypes)
def test_randomuniformlike_stats(op_tester, dtypes):
    expected_min = np.array(-1.0, dtype=dtypes[0])
    expected_max = np.array(1.0, dtype=dtypes[0])
    two = np.array(2.0, dtype=dtypes[0])
    expected_mean = (expected_max + expected_min) / two
    data = np.zeros((10000, 1), dtype=dtypes[0])

    def init_builder(builder):
        T = builder.addInputTensor(data)
        ru = builder.aiOnnx.randomuniformlike([T],
                                              dtype=dtypes[1],
                                              low=expected_min,
                                              high=expected_max)

        actual_min = builder.aiOnnx.reducemin([ru], keepdims=False)
        builder.addOutputTensor(actual_min)

        actual_max = builder.aiOnnx.reducemax([ru], keepdims=False)
        builder.addOutputTensor(actual_max)

        actual_mean = builder.aiOnnx.reducemean([ru], keepdims=False)
        builder.addOutputTensor(actual_mean)

        return [actual_min, actual_max, actual_mean]

    def reference(ref_data):
        return [expected_min, expected_max, expected_mean]

    op_tester.atol = 1e-02
    op_tester.run(init_builder, reference, seed=8)
