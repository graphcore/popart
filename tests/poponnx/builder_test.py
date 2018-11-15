import pytest

import poponnx


def test_basic():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    o = builder.add(i1, i2)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    with pytest.raises(TypeError) as e_info:
        builder.add(0, 0)

    assert (e_info.value.args[0].startswith("add(): incompatible function"))


def test_add_conv():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution(i1, i2, [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    with pytest.raises(TypeError) as e_info:
        builder.convolution()

    assert (e_info.value.args[0].startswith(
        "convolution(): incompatible function"))


def test_add_conv_and_bias():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))
    i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4]))

    o = builder.convolutionWithBias(i1, i2, i3, [1, 1], [0, 0, 0, 0], [1, 1],
                                    1)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(i3) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    with pytest.raises(TypeError) as e_info:
        builder.convolutionWithBias()

    assert (
        e_info.value.args[0].startswith("convolutionWithBias(): incompatible"))


def test_add_gemm():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [12, 8]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [8, 16]))
    i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [16]))

    o = builder.gemm(i1, i2, i3, 1., 1., 0, 0)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    with pytest.raises(TypeError) as e_info:
        builder.gemm(0, 0, 0, 0, 0, 0, 0)

    assert (e_info.value.args[0].startswith("gemm(): incompatible function"))


def test_add_matmul():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [2, 3]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [3, 4]))

    o = builder.matmul(i1, i2)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    with pytest.raises(TypeError) as e_info:
        builder.matmul(0, 0, 0, 0, 0, 0, 0)

    assert (e_info.value.args[0].startswith("matmul(): incompatible function"))
