# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import os
import popart
import pytest
import tempfile
import test_util as tu


def getDevice():

    # A cpu device
    return tu.create_test_device()


def test_basic():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))

    old_o = ""

    o = builder.aiOnnx.abs([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.acos([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.acosh([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.add([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.logical_and([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.asin([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.asinh([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.atan([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.atanh([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.ceil([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.cos([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.cosh([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.div([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.elu([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.equal([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.exp([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.floor([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.greater([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.identity([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.less([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.log([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.logsoftmax([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.max([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.mean([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.min([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.mul([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.neg([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.logical_not([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.logical_or([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.pow([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.reciprocal([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.relu([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.sigmoid([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.sin([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.sinh([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.softsign([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.softmax([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.sqrt([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.sub([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.sum([i1, i2])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.tan([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.tanh([i1])
    assert (old_o != o)
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    old_o = o

    o = builder.aiOnnx.logical_xor([i1, i2])
    assert (builder.getTensorShape(o) == [1, 2, 32, 32])
    assert (old_o != o)

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.abs([])
    assert (e_info.value.args[0].startswith("Abs has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.acos([])
    assert (e_info.value.args[0].startswith("Acos has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.acosh([])
    assert (e_info.value.args[0].startswith("Acosh has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.add([])
    assert (e_info.value.args[0].startswith("Add has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.logical_and([])
    assert (e_info.value.args[0].startswith("And has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.asin([])
    assert (e_info.value.args[0].startswith("Asin has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.asinh([])
    assert (e_info.value.args[0].startswith("Asinh has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.atan([])
    assert (e_info.value.args[0].startswith("Atan has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.atanh([])
    assert (e_info.value.args[0].startswith("Atanh has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.ceil([])
    assert (e_info.value.args[0].startswith("Ceil has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.cos([])
    assert (e_info.value.args[0].startswith("Cos has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.cosh([])
    assert (e_info.value.args[0].startswith("Cosh has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.div([])
    assert (e_info.value.args[0].startswith("Div has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.elu([])
    assert (e_info.value.args[0].startswith("Elu has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.equal([])
    assert (e_info.value.args[0].startswith("Equal has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.exp([])
    assert (e_info.value.args[0].startswith("Exp has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.floor([])
    assert (e_info.value.args[0].startswith("Floor has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.greater([])
    assert (e_info.value.args[0].startswith("Greater has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.identity([])
    assert (e_info.value.args[0].startswith("Identity has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.less([])
    assert (e_info.value.args[0].startswith("Less has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.log([])
    assert (e_info.value.args[0].startswith("Log has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.max([])
    assert (e_info.value.args[0].startswith("Max has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.mean([])
    assert (e_info.value.args[0].startswith("Mean has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.min([])
    assert (e_info.value.args[0].startswith("Min has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.mul([])
    assert (e_info.value.args[0].startswith("Mul has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.neg([])
    assert (e_info.value.args[0].startswith("Neg has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.logical_not([])
    assert (e_info.value.args[0].startswith("Not has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.logical_or([])
    assert (e_info.value.args[0].startswith("Or has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.pow([])
    assert (e_info.value.args[0].startswith("Pow has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.reciprocal([])
    assert (
        e_info.value.args[0].startswith("Reciprocal has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.relu([])
    assert (e_info.value.args[0].startswith("Relu has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.sigmoid([])
    assert (e_info.value.args[0].startswith("Sigmoid has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.sin([])
    assert (e_info.value.args[0].startswith("Sin has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.sinh([])
    assert (e_info.value.args[0].startswith("Sinh has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.softsign([])
    assert (e_info.value.args[0].startswith("Softsign has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.sqrt([])
    assert (e_info.value.args[0].startswith("Sqrt has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.sub([])
    assert (e_info.value.args[0].startswith("Sub has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.sum([])
    assert (e_info.value.args[0].startswith("Sum has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.tan([])
    assert (e_info.value.args[0].startswith("Tan has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.tanh([])
    assert (e_info.value.args[0].startswith("Tanh has invalid number of"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.logical_xor([])
    assert (e_info.value.args[0].startswith("Xor has invalid number of"))

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    with pytest.raises(TypeError) as e_info:
        builder.aiOnnx.add(0, 0)

    assert (e_info.value.args[0].startswith("add(): incompatible function"))


def test_add_pad():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [10]))

    o = builder.aiOnnx.pad([i1], [2, 3], "constant", 0.0)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)

    assert (builder.getTensorShape(o) == [15])

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.pad([i1], [2, 3, 4, 5], "constant", 0.0)

    assert (e_info.value.args[0].startswith("Padding vector (length 4) "))


def test_add_subsample():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [10, 9, 8, 7]))

    o = builder.aiGraphcore.subsample([i1], strides=[2, 3, 4, 5])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)

    assert (builder.getTensorShape(o) == [5, 3, 2, 1])


def test_add_constant():

    builder = popart.Builder()

    c = builder.aiOnnx.constant(np.array([[1, 6], [4, 5]], dtype=np.float32))

    builder.addOutputTensor(c)

    proto = builder.getModelProto()

    assert (len(proto) > 0)

    assert (builder.getTensorShape(c) == [2, 2])


def test_add_conv():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    assert (builder.getTensorShape(o) == [1, 4, 30, 30])

    with pytest.raises(TypeError) as e_info:
        builder.aiOnnx.conv()

    assert (e_info.value.args[0].startswith("conv(): incompatible function"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1])

    assert (e_info.value.args[0].startswith(
        "Length of strides vector 1 != number"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0],
                            strides=[1, 1])

    assert (e_info.value.args[0].startswith(
        "Padding vector (length 2) does not have 2 values for each spatial"))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.conv([i1, i2],
                            dilations=[1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    assert (e_info.value.args[0].startswith(
        "Length of dilations vector 2 != number of spatial d"))


def test_add_conv_and_bias():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4]))

    o = builder.aiOnnx.conv([i1, i2, i3],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(i3) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    assert (builder.getTensorShape(o) == [1, 4, 30, 30])

    with pytest.raises(TypeError) as e_info:
        builder.aiOnnx.conv()

    assert (e_info.value.args[0].startswith("conv(): incompatible"))


def test_add_gemm():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [12, 8]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [8, 16]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [16]))

    o = builder.aiOnnx.gemm([i1, i2, i3], 1., 1., 0, 0)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    assert (builder.getTensorShape(o) == [12, 16])

    with pytest.raises(TypeError) as e_info:
        builder.aiOnnx.gemm(0, 0, 0, 0, 0, 0, 0)

    assert (e_info.value.args[0].startswith("gemm(): incompatible function"))


def test_add_matmul():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [3, 4]))

    o = builder.aiOnnx.matmul([i1, i2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert (len(proto) > 0)
    assert (len(i1) > 0)
    assert (len(i2) > 0)
    assert (len(o) > 0)
    assert (i1 != i2)
    assert (i2 != o)

    assert (builder.getTensorShape(o) == [2, 4])

    with pytest.raises(TypeError) as e_info:
        builder.aiOnnx.matmul(0, 0, 0, 0, 0, 0, 0)

    assert (e_info.value.args[0].startswith("matmul(): incompatible function"))


def test_add_reshape():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]))
    i2 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))

    o = builder.aiOnnx.reshape([i1, i2])

    builder.addOutputTensor(o)

    assert (builder.getTensorShape(o) == [1, 6])


def test_add_reshape_const():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]))

    o = builder.reshape_const(builder.aiOnnx, [i1], [1, 6])

    builder.addOutputTensor(o)

    assert (builder.getTensorShape(o) == [1, 6])


def test_initialized_input_fp32():

    builder1 = popart.Builder()
    builder1.addInputTensor(popart.TensorInfo("FLOAT", [100, 100]))
    proto1 = builder1.getModelProto()
    assert (len(proto1) > 0)

    builder2 = popart.Builder()
    builder2.addInitializedInputTensor(np.ones([100, 100], np.float32))
    proto2 = builder2.getModelProto()

    assert (len(proto2) > 0)

    assert (len(proto2) > len(proto1))


def test_initialized_input_int32():

    builder1 = popart.Builder()
    builder1.addInputTensor(popart.TensorInfo("INT32", [100, 100]))
    proto1 = builder1.getModelProto()
    assert (len(proto1) > 0)

    builder2 = popart.Builder()
    builder2.addInitializedInputTensor(np.ones([100, 100], np.int32))
    proto2 = builder2.getModelProto()

    assert (len(proto2) > 0)

    assert (len(proto2) > len(proto1))


def test_initialized_input_fp16():

    builder1 = popart.Builder()
    builder1.addInputTensor(popart.TensorInfo("FLOAT16", [100, 100]))
    proto1 = builder1.getModelProto()
    assert (len(proto1) > 0)

    builder2 = popart.Builder()
    builder2.addInitializedInputTensor(np.ones([100, 100], np.float16))
    proto2 = builder2.getModelProto()

    assert (len(proto2) > 0)

    assert (len(proto2) > len(proto1))


def test_initialized_input_bool():

    builder1 = popart.Builder()
    builder1.addInputTensor(popart.TensorInfo("BOOL", [100, 100]))
    proto1 = builder1.getModelProto()
    assert (len(proto1) > 0)

    builder2 = popart.Builder()
    builder2.addInitializedInputTensor(np.ones([100, 100], np.bool))
    proto2 = builder2.getModelProto()

    assert (len(proto2) > 0)

    assert (len(proto2) > len(proto1))


def test_inout_tensor_info():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    x = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])
    y = builder.aiOnnx.conv([i1, i3],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])
    o = builder.aiOnnx.add([x, y])

    builder.addOutputTensor(o)

    assert (builder.getInputTensorIds() == [i1, i2, i3])
    assert (builder.getOutputTensorIds() == [o])
    assert (builder.getValueTensorIds() == [x, y, o])
    assert (builder.getTensorShape(i1) == [1, 2, 32, 32])
    assert (builder.getTensorShape(i2) == [4, 2, 3, 3])
    assert (builder.getTensorShape(i3) == [4, 2, 3, 3])
    assert (builder.getTensorShape(x) == [1, 4, 30, 30])
    assert (builder.getTensorShape(y) == [1, 4, 30, 30])
    assert (builder.getTensorShape(o) == [1, 4, 30, 30])

    with pytest.raises(popart.popart_exception) as e_info:
        builder.getTensorShape("NotAnId")
    assert (
        e_info.value.args[0].find("is not an input or output tensor. Must be"))


def test_set_virtual_graph():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])
    builder.virtualGraph(o, 1)

    builder.addOutputTensor(o)
    # Test that tha attribute has been set
    val = 1
    res = builder.getVirtualGraph(o)
    assert (res == val)


def test_set_weights_from_host():

    # Run the first builder
    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)

    data = np.array([1, 2], dtype=np.float32)

    i2 = builder.addInitializedInputTensor(data)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    optimizer = popart.ConstSGD(0.01)
    losses = [popart.L1Loss(o, "l1LossVal", 0.1)]

    session = popart.TrainingSession(fnModel=proto,
                                     dataFeed=dataFlow,
                                     losses=losses,
                                     optimizer=optimizer,
                                     deviceInfo=getDevice())

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {i1: np.array([1, 2], dtype=np.float32)}
    stepio = popart.PyStepIO(inputs, anchors)

    with pytest.raises(popart.popart_exception) as e_info:
        session.run(stepio)

    assert (e_info.value.args[0].startswith(
        "Must call weightsFromHost before run as the"))


def test_add_int_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = 100
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.getInt64NodeAttribute("test", set([o]))
    assert (res == val)


def test_add_int_vector_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = [100, 200, -1]
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.getInt64VectorNodeAttribute("test", set([o]))
    assert (res == val)


def test_add_float_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = .1
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.getFloatNodeAttribute("test", set([o]))
    assert (res == pytest.approx(val))


def test_add_float_vector_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = [100., -.1, 100.0, 10.0]
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.getFloatVectorNodeAttribute("test", set([o]))
    assert (res == pytest.approx(val))


def test_add_string_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = "test"
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.getStringNodeAttribute("test", set([o]))
    assert (res == val)


def test_add_string_vector_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = ["test", "test2", "test"]
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.getStringVectorNodeAttribute("test", set([o]))
    assert (res == val)


def test_add_attribute_missing_node():

    builder = popart.Builder()
    val = 100
    with pytest.raises(popart.popart_exception) as e_info:
        builder.addNodeAttribute("test", val, set(("i", "j")))

    assert (
        e_info.value.args[0].find("Could not find a node with outputs i, j."))


def test_has_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = 100
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.nodeHasAttribute("test", set([o]))
    assert (res)
    res = builder.nodeHasAttribute("test2", set([o]))
    assert (not res)


def test_get_all_attribute_names():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set([o]))
    builder.addNodeAttribute("test2", -1, set([o]))
    builder.addNodeAttribute("test3", "abba", set([o]))
    res = builder.getAllNodeAttributeNames(set([o]))
    assert (set(res) == set(("test", "test2", "test3")))

    res = builder.getFloatNodeAttribute("test", set([o]))
    assert (res == pytest.approx(100.))

    res = builder.getInt64NodeAttribute("test2", set([o]))
    assert (res == -1)

    res = builder.getStringNodeAttribute("test3", set([o]))
    assert (res == "abba")


def test_get_conv_strides_attribute():

    strides = [1, 1]

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=strides)

    builder.addOutputTensor(o)
    res = builder.getInt64VectorNodeAttribute("strides", set([o]))
    assert (res == strides)


def test_dont_override_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set([o]))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.addNodeAttribute("test", 100, set([o]))

    assert (e_info.value.args[0] == "Node already has attribute test.")


def test_get_attribute_doesnt_exist():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set([o]))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.getFloatNodeAttribute("test1", set([o]))

    assert (e_info.value.args[0] == "Node does not have an attribute test1.")


def test_remove_attribute():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = 100
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.nodeHasAttribute("test", set([o]))
    assert (res)
    builder.removeNodeAttribute("test", set([o]))
    res = builder.nodeHasAttribute("test", set([o]))
    assert (not res)


def test_remove_attribute_doesnt_exist():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.aiOnnx.conv([i1, i2],
                            dilations=[1, 1],
                            pads=[0, 0, 0, 0],
                            strides=[1, 1])

    builder.addOutputTensor(o)
    # Set then get
    val = 100
    builder.addNodeAttribute("test", val, set([o]))
    res = builder.nodeHasAttribute("test", set([o]))
    assert (res)
    with pytest.raises(popart.popart_exception) as e_info:
        builder.removeNodeAttribute("test1", set([o]))

    assert (e_info.value.args[0] ==
            "Cannot remove attribute test1 as it does not exist.")


def test_get_attribute_wrong_type_int():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set([o]))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.getInt64NodeAttribute("test", set([o]))

    assert (e_info.value.args[0].find("Node test is not an integer."))


def test_get_attribute_wrong_type_int_vector():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set([o]))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.getInt64VectorNodeAttribute("test", set([o]))

    assert (e_info.value.args[0].find("Node test is not an integer vector."))


def test_get_attribute_wrong_type_float():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100, set([o]))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.getFloatNodeAttribute("test", set([o]))

    assert (e_info.value.args[0].find("Node test is not a float."))


def test_get_attribute_wrong_type_float_vector():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100, set([o]))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.getFloatVectorNodeAttribute("test", set([o]))

    assert (e_info.value.args[0].find("Node test is not a float vector."))


def test_get_attribute_wrong_type_string():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100, set([o]))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.getStringNodeAttribute("test", set([o]))

    assert (e_info.value.args[0].find("Node test is not a string."))


def test_get_attribute_wrong_type_string_vector():

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.aiOnnx.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100, set([o]))

    with pytest.raises(popart.popart_exception) as e_info:
        builder.getStringVectorNodeAttribute("test", set([o]))

    assert (e_info.value.args[0].find("Node test is not a string vector."))


def test_load_onnx_model_from_other_builder(tmpdir):

    # Run the first builder
    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFeed=dataFlow,
                                      deviceInfo=getDevice())

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1: np.array([1, 2], dtype=np.float32),
        i2: np.array([3, 4], dtype=np.float32)
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)
    assert (np.array_equal(anchors[o], [4, 6]))

    # Run a builder that imports the model of the other builder and check the
    # output is still the same
    builder2 = popart.Builder(proto)

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    proto2 = builder.getModelProto()
    session = popart.InferenceSession(fnModel=proto2,
                                      dataFeed=dataFlow,
                                      deviceInfo=getDevice())

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1: np.array([1, 2], dtype=np.float32),
        i2: np.array([3, 4], dtype=np.float32)
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    assert (np.array_equal(anchors[o], [4, 6]))


def test_load_onnx_model_from_file(tmpdir):

    # Create a builder, store the model in a file and load it into a different
    # builder.
    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)
    filename = tmpdir + "/model.onnx"
    with open(filename, 'wb') as out:
        out.write(builder.getModelProto())

    builder2 = popart.Builder(str(filename))

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})
    optimizer = popart.ConstSGD(0.01)
    losses = [popart.L1Loss(o, "l1LossVal", 0.1)]

    proto = builder2.getModelProto()

    session = popart.InferenceSession(fnModel=proto,
                                      dataFeed=dataFlow,
                                      deviceInfo=getDevice())

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1: np.array([1, 2], dtype=np.float32),
        i2: np.array([3, 4], dtype=np.float32)
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    assert (np.array_equal(anchors[o], [4, 6]))


def test_convert_initializers_to_constants(tmpdir):
    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]))
    i2 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))

    o = builder.aiOnnx.reshape([i1, i2])

    builder.addOutputTensor(o)

    graph_transformer = popart.GraphTransformer(builder.getModelProto())
    graph_transformer.convertInitializersToConstants([i2])

    with pytest.raises(popart.popart_exception) as e_info:
        graph_transformer.convertInitializersToConstants(["unknown"])
    assert (e_info.value.args[0] ==
            "TensorId unknown not in the model initalizers")

    builder = popart.Builder(graph_transformer.getModelProto())

    ids = builder.getInputTensorIds()
    assert (i1 in ids)
    assert (i2 not in ids)


def test_convert_all_fixed_point_initializers_to_constants(tmpdir):
    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]))
    i2 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))
    o1 = builder.aiOnnx.reshape([i1, i2])

    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", [3, 2]))
    i4 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))
    o2 = builder.aiOnnx.reshape([i3, i4])

    o = builder.aiOnnx.add([o1, o2])
    builder.addOutputTensor(o)

    graph_transformer = popart.GraphTransformer(builder.getModelProto())
    graph_transformer.convertAllFixedPointInitializersToConstants()

    builder = popart.Builder(graph_transformer.getModelProto())

    ids = builder.getInputTensorIds()
    assert (i1 in ids)
    assert (i2 not in ids)

    assert (i3 in ids)
    assert (i4 not in ids)


def test_builder_opsetDefaultVersions(tmpdir):

    # This will create a build with the default opsets
    # This test may fail when we upgrade ops....
    builder = popart.Builder()

    i1 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))
    i2 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))

    # This will work as it will use opset 9 version of add
    o1 = builder.aiOnnx.add([i1, i2], 'add_v9')

    # This will not work as it will use opset 6 version of add
    with pytest.raises(TypeError) as e_info:
        o1 = builder.aiOnnx.add([i1, i2], 1, 0, 'add_v6')
    assert (
        "add(): incompatible function arguments. The following argument types are supported:"
        in e_info.value.args[0])


def test_builder_opsetDefinesVersions(tmpdir):

    # This will create a build with the default opsets
    # This test may fail when we upgrade ops....
    builder = popart.Builder(opsets={"ai.onnx": 6, "ai.graphcore": 1})

    i1 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))
    i2 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))

    # This will work as it will use opset 6 version of add
    o1 = builder.aiOnnx.add([i1, i2], 1, 0, 'add_v6')

    # This will not work as it will use opset 9 version of add
    with pytest.raises(TypeError) as e_info:
        o1 = builder.aiOnnx.add([i1, i2], 'add_v9')
    assert (
        "add(): incompatible function arguments. The following argument types are supported:"
        in e_info.value.args[0])


def test_builder_opsetVersioning(tmpdir):

    builder = popart.Builder(opsets={"ai.onnx": 9, "ai.graphcore": 1})

    i1 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))
    i2 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))

    o1 = builder.aiOnnx.add([i1, i2], 'b')

    o1 = builder.aiOnnxOpset9.add([i1, i2], 'a')

    # This will fail as we have already defined the opset as 9
    with pytest.raises(popart.popart_exception) as e_info:
        o1 = builder.aiOnnxOpset6.add([i1, i2], 1, 0, 'a')
    assert (
        "Invalid opset 6 used to add an operation. Opset for domain ai.onnx already defined as 9"
        in e_info.value.args[0])


def test_name_scope(tmpdir):
    builder = popart.Builder(opsets={"ai.onnx": 9, "ai.graphcore": 1})

    i1 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))
    i2 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))

    with builder.nameScope('scope1'):
        o1 = builder.aiOnnx.add([i1, i2], 'myop1')

    with builder.nameScope('scope2'):
        o2 = builder.aiOnnx.add([i1, i2], 'myop2')

        with builder.nameScope('embedded_scope'):
            o3 = builder.aiOnnx.add([o1, o2], 'myop3')

    assert (o1.startswith('scope1'))
    assert (o2.startswith('scope2'))
    assert (o3.startswith('scope2/embedded_scope'))


def test_tensor_names(tmpdir):
    builder = popart.Builder(opsets={"ai.onnx": 9, "ai.graphcore": 1})

    i1 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))
    i2 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64),
                                           "weights")

    assert (i1 == "init_input")
    assert (i2 == "weights")

    with builder.nameScope("layer1"):
        i3 = builder.addInitializedInputTensor(np.array([1, 6],
                                                        dtype=np.int64))
        i4 = builder.addInitializedInputTensor(
            np.array([1, 6], dtype=np.int64), "weights")

    assert (i3 == "layer1/init_input")
    assert (i4 == "layer1/weights")

    i5 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]))
    i6 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]), "data")

    with builder.nameScope("layer2"):
        i7 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]),
                                    "label")

    assert (i5 == "input")
    assert (i6 == "data")
    assert (i7 == "layer2/label")

    i10 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]))
    i11 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 3]), "data")

    assert (i10 == "input/1")
    assert (i11 == "data/1")


def test_add_untyped_input():
    builder = popart.Builder()

    # Trying to add an untyped input tensor on the top-level
    # graph should throw an error.
    with pytest.raises(popart.popart_exception) as e_info:
        builder.addUntypedInputTensor()

    assert e_info.value.args[0].startswith(
        "Can not add untyped tensors to the top-level graph.")

    # Adding an untyped input on a subgraph should be fine.
    sub_builder = builder.createSubgraphBuilder()
    x = sub_builder.addUntypedInputTensor()


def test_save_model_to_file():
    builder = popart.Builder()
    i = builder.addInputTensor(popart.TensorInfo("FLOAT", [2]))
    o = builder.aiOnnx.sqrt([i])

    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, "model.onnx")
    builder.saveModelProto(tmpfile)

    # Check file has saved
    assert os.path.exists(tmpfile) == True

    # Check that, when re-loaded, its contents are the same as when saved
    new_builder = popart.Builder(tmpfile)
    assert new_builder.getModelProto() == builder.getModelProto()


def test_get_tensor_type():
    builder = popart.Builder()
    i0 = builder.addInputTensor(popart.TensorInfo("FLOAT", [10, 9, 8, 7]))
    act0 = builder.aiGraphcore.subsample([i0], strides=[2, 3, 4, 5])
    act1 = builder.aiOnnx.cast([act0], "INT32")
    assert builder.getTensorDtypeString(i0) == "float32"
    assert builder.getTensorDtypeString(act0) == "float32"
    assert builder.getTensorDtypeString(act1) == "int32"


def test_is_initializer():
    builder = popart.Builder()
    i0 = builder.addInputTensor(popart.TensorInfo("FLOAT", [10, 9, 8, 7]))
    i1 = builder.addInitializedInputTensor(np.array([1, 6], dtype=np.int64))
    assert builder.isInitializer(i0) == False
    assert builder.isInitializer(i1) == True


def test_conv_kernel_shape_mismatch():
    builder = popart.Builder()
    i = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 64, 56, 56]))
    w = builder.addInputTensor(popart.TensorInfo("FLOAT", [256, 64, 1, 1]))
    # kernel shape, as inferred from weights shape = [1, 1]
    with pytest.raises(popart.popart_exception) as e_info:
        o = builder.aiOnnx.conv(
            [i, w],
            dilations=[1, 1],
            kernel_shape=[64, 64],  # not [1, 1]
            strides=[1, 1],
            pads=[0, 0, 0, 0])
    assert (
        e_info.value.args[0] ==
        "kernel_shape, [64 64], does not match inferred shape from weight input '"
        + w + "', [1 1]")


def test_conv_invalid_kernel_shape():
    builder = popart.Builder()
    i = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 64, 56, 56]))
    w = builder.addInputTensor(popart.TensorInfo("FLOAT", [256, 64, 64, 64]))
    with pytest.raises(popart.popart_exception) as e_info:
        # Kernel [64, 64] not valid for spatial input [56, 56], given other
        # conv window parameters
        o = builder.aiOnnx.conv([i, w],
                                dilations=[1, 1],
                                strides=[1, 1],
                                pads=[0, 0, 0, 0])
    assert (
        e_info.value.args[0] ==
        "Window parameter values combine to give invalid spatial output shape: [-7 -7]"
    )
