import pytest

import poponnx


def test_basic():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    old_o = ""

    o = builder.abs([i1])
    assert (old_o != o)
    old_o = o

    o = builder.acos([i1])
    assert (old_o != o)
    old_o = o

    o = builder.acosh([i1])
    assert (old_o != o)
    old_o = o

    o = builder.add([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.logical_and([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.asin([i1])
    assert (old_o != o)
    old_o = o

    o = builder.asinh([i1])
    assert (old_o != o)
    old_o = o

    o = builder.atan([i1])
    assert (old_o != o)
    old_o = o

    o = builder.atanh([i1])
    assert (old_o != o)
    old_o = o

    o = builder.ceil([i1])
    assert (old_o != o)
    old_o = o

    o = builder.cos([i1])
    assert (old_o != o)
    old_o = o

    o = builder.cosh([i1])
    assert (old_o != o)
    old_o = o

    o = builder.div([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.elu([i1])
    assert (old_o != o)
    old_o = o

    o = builder.equal([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.exp([i1])
    assert (old_o != o)
    old_o = o

    o = builder.floor([i1])
    assert (old_o != o)
    old_o = o

    o = builder.greater([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.identity([i1])
    assert (old_o != o)
    old_o = o

    o = builder.less([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.log([i1])
    assert (old_o != o)
    old_o = o

    o = builder.max([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.mean([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.min([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.mul([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.neg([i1])
    assert (old_o != o)
    old_o = o

    o = builder.logical_not([i1])
    assert (old_o != o)
    old_o = o

    o = builder.logical_or([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.pow([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.reciprocal([i1])
    assert (old_o != o)
    old_o = o

    o = builder.relu([i1])
    assert (old_o != o)
    old_o = o

    o = builder.sigmoid([i1])
    assert (old_o != o)
    old_o = o

    o = builder.sin([i1])
    assert (old_o != o)
    old_o = o

    o = builder.sinh([i1])
    assert (old_o != o)
    old_o = o

    o = builder.softsign([i1])
    assert (old_o != o)
    old_o = o

    o = builder.sqrt([i1])
    assert (old_o != o)
    old_o = o

    o = builder.sub([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.sum([i1, i2])
    assert (old_o != o)
    old_o = o

    o = builder.tan([i1])
    assert (old_o != o)
    old_o = o

    o = builder.tanh([i1])
    assert (old_o != o)
    old_o = o

    o = builder.logical_xor([i1, i2])
    assert (old_o != o)

    with pytest.raises(poponnx.exception) as e_info:
        builder.abs([])
    assert (e_info.value.args[0].startswith("Abs has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.acos([])
    assert (e_info.value.args[0].startswith("Acos has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.acosh([])
    assert (e_info.value.args[0].startswith("Acosh has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.add([])
    assert (e_info.value.args[0].startswith("Add has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.logical_and([])
    assert (e_info.value.args[0].startswith("And has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.asin([])
    assert (e_info.value.args[0].startswith("Asin has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.asinh([])
    assert (e_info.value.args[0].startswith("Asinh has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.atan([])
    assert (e_info.value.args[0].startswith("Atan has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.atanh([])
    assert (e_info.value.args[0].startswith("Atanh has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.ceil([])
    assert (e_info.value.args[0].startswith("Ceil has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.cos([])
    assert (e_info.value.args[0].startswith("Cos has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.cosh([])
    assert (e_info.value.args[0].startswith("Cosh has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.div([])
    assert (e_info.value.args[0].startswith("Div has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.elu([])
    assert (e_info.value.args[0].startswith("Elu has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.equal([])
    assert (e_info.value.args[0].startswith("Equal has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.exp([])
    assert (e_info.value.args[0].startswith("Exp has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.floor([])
    assert (e_info.value.args[0].startswith("Floor has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.greater([])
    assert (e_info.value.args[0].startswith("Greater has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.identity([])
    assert (e_info.value.args[0].startswith("Identity has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.less([])
    assert (e_info.value.args[0].startswith("Less has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.log([])
    assert (e_info.value.args[0].startswith("Log has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.max([])
    assert (e_info.value.args[0].startswith("Max has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.mean([])
    assert (e_info.value.args[0].startswith("Mean has no arguments"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.min([])
    assert (e_info.value.args[0].startswith("Min has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.mul([])
    assert (e_info.value.args[0].startswith("Mul has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.neg([])
    assert (e_info.value.args[0].startswith("Neg has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.logical_not([])
    assert (e_info.value.args[0].startswith("Not has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.logical_or([])
    assert (e_info.value.args[0].startswith("Or has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.pow([])
    assert (e_info.value.args[0].startswith("Pow has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.reciprocal([])
    assert (
        e_info.value.args[0].startswith("Reciprocal has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.relu([])
    assert (e_info.value.args[0].startswith("Relu has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.sigmoid([])
    assert (e_info.value.args[0].startswith("Sigmoid has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.sin([])
    assert (e_info.value.args[0].startswith("Sin has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.sinh([])
    assert (e_info.value.args[0].startswith("Sinh has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.softsign([])
    assert (e_info.value.args[0].startswith("Softsign has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.sqrt([])
    assert (e_info.value.args[0].startswith("Sqrt has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.sub([])
    assert (e_info.value.args[0].startswith("Sub has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.sum([])
    assert (e_info.value.args[0].startswith("Sum has no arguments"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.tan([])
    assert (e_info.value.args[0].startswith("Tan has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.tanh([])
    assert (e_info.value.args[0].startswith("Tanh has invalid number of"))

    with pytest.raises(poponnx.exception) as e_info:
        builder.logical_xor([])
    assert (e_info.value.args[0].startswith("Xor has invalid number of"))

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

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

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

    o = builder.convolution([i1, i2, i3], [1, 1], [0, 0, 0, 0], [1, 1], 1)

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
        builder.convolution()

    assert (e_info.value.args[0].startswith("convolution(): incompatible"))


def test_add_gemm():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [12, 8]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [8, 16]))
    i3 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [16]))

    o = builder.gemm([i1, i2, i3], 1., 1., 0, 0)

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

    o = builder.matmul([i1, i2])

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
