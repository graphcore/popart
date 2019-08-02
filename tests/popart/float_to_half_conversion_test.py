import numpy as np
import pytest
import popart
import torch
import test_util as tu


def test_conversion_basic():
    data = np.array([[1, 2]])

    builder = popart.Builder()

    inInfo = popart.TensorInfo("FLOAT", data.shape)

    i1 = builder.addInputTensor(inInfo)
    o = builder.aiOnnx.identity([i1])
    builder.addOutputTensor(o)

    float_proto = builder.getModelProto()
    half_proto = _convert_floats_to_halfs(float_proto)

    inputs = {i1: data}
    float_anchors = _run_model(float_proto, _as_type(inputs, np.float32), o)
    half_anchors = _run_model(half_proto, _as_type(inputs, np.float16), o)

    _check_anchors(float_anchors, half_anchors)


def test_conversion_with_mul():
    d1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    d2 = np.array([[0.1, 0.4, 0.2, 0.5, 0.3, 0.6, 0.4, 0.7]])

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", d1.shape))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", d2.shape))
    o = builder.aiOnnx.mul([i1, i2])
    builder.addOutputTensor(o)

    float_proto = builder.getModelProto()
    half_proto = _convert_floats_to_halfs(float_proto)

    inputs = {i1: d1, i2: d2}
    float_anchors = _run_model(float_proto, _as_type(inputs, np.float32), o)
    half_anchors = _run_model(half_proto, _as_type(inputs, np.float16), o)

    _check_anchors(float_anchors, half_anchors)

    np_half_o = d1.astype(np.float16) * d2.astype(np.float16)
    print(half_anchors[o])
    print(np_half_o)


def test_conversion_with_initializers():
    d1 = np.array([[0.1, 0.4, 0.2, 0.5, 0.3, 0.6, 0.4, 0.7]])
    # need to provide some input otherwise const folding errors
    d2 = np.zeros(d1.shape)

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", d2.shape))
    c = builder.aiOnnx.constant(d1.astype(np.float32))
    o = builder.aiOnnx.add([i1, c])
    builder.addOutputTensor(o)

    float_proto = builder.getModelProto()
    half_proto = _convert_floats_to_halfs(float_proto)

    inputs = {i1: d2}
    float_anchors = _run_model(float_proto, _as_type(inputs, np.float32), o)
    half_anchors = _run_model(half_proto, _as_type(inputs, np.float16), o)

    _check_anchors(float_anchors, half_anchors)


# This test demonstrates the rtol required for testing
# the same operation using 32 and 16 bit floats
def test_conversions_with_mul_numpy():
    d1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    d2 = np.array([[0.1, 0.4, 0.2, 0.5, 0.3, 0.6, 0.4, 0.7]])

    o_float = d1.astype(np.float32) * d2.astype(np.float32)
    o_half = d1.astype(np.float16) * d2.astype(np.float16)

    assert not np.allclose(o_float, o_half, rtol=1e-04)
    assert np.allclose(o_float, o_half, rtol=1e-03)


def _check_anchors(float_anchors, half_anchors):
    for key in float_anchors.keys():
        print("Testing anchor '{key}'")
        # using rtol 1e-03 because of the comparison of float32 and float16
        # see `test_conversion_with_mul_numpy`
        assert np.allclose(float_anchors[key], half_anchors[key], rtol=1e-03)


def _run_model(model, inputs, output, batchesPerStep=1):
    dataFlow = popart.DataFlow(batchesPerStep,
                               {output: popart.AnchorReturnType("ALL")})
    session = popart.InferenceSession(fnModel=model,
                                      dataFeed=dataFlow,
                                      deviceInfo=tu.get_poplar_cpu_device())

    session.prepareDevice()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO(inputs, anchors)
    session.run(stepio)

    return anchors


def _as_type(d, dtype):
    return {k: v.astype(dtype) for k, v in d.items()}


def _convert_floats_to_halfs(proto):
    graph_transformer = popart.GraphTransformer(proto)
    graph_transformer.convertFloatsToHalfs()
    return graph_transformer.getModelProto()
