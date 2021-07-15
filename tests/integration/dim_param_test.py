# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import pytest
import popart
import onnx

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu
from operators_test.op_tester import op_tester


def get_model_with_dim_param(tmpdir):
    # build the model
    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]),
                                "input_0")
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]),
                                "input_1")
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    # insert a dim_param into the model
    model = onnx.load_from_string(proto)
    graph = model.graph
    inputs = graph.input
    inputs[0].type.tensor_type.shape.dim[0].dim_param = 'fubar'
    edited_model = tmpdir / 'test_model.onnx'
    onnx.save(model, str(edited_model))
    with open(edited_model, 'rb') as f:
        proto = f.read()

    return proto, o


def test_model_with_unspecified_dim_params(tmpdir):
    proto, outId = get_model_with_dim_param(tmpdir)

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=proto,
            dataFlow=popart.DataFlow(1,
                                     {outId: popart.AnchorReturnType("All")}),
            deviceInfo=tu.create_test_device())

    assert e_info.value.args[0] == (
        "Input tensor 'input_0' must be specified in InputShapeInfo, as it has shape [fubar, 2, 32, 32], which uses an unknown value 'fubar'."
    )


def test_model_with_specified_dim_params(tmpdir):
    proto, outId = get_model_with_dim_param(tmpdir)

    inputShapeInfo = popart.InputShapeInfo()
    inputShapeInfo.add("input_0", popart.TensorInfo("FLOAT", [1, 2, 32, 32]))

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=popart.DataFlow(1, {outId: popart.AnchorReturnType("All")}),
        deviceInfo=tu.create_test_device(),
        inputShapeInfo=inputShapeInfo)
