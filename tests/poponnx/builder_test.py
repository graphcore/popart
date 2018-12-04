import numpy as np
import pytest
import poponnx
import pytest


def getDevice():

    # A cpu device
    return poponnx.DeviceManager().createCpuDevice()


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

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.abs([])
    assert (e_info.value.args[0].startswith("Abs has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.acos([])
    assert (e_info.value.args[0].startswith("Acos has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.acosh([])
    assert (e_info.value.args[0].startswith("Acosh has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.add([])
    assert (e_info.value.args[0].startswith("Add has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.logical_and([])
    assert (e_info.value.args[0].startswith("And has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.asin([])
    assert (e_info.value.args[0].startswith("Asin has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.asinh([])
    assert (e_info.value.args[0].startswith("Asinh has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.atan([])
    assert (e_info.value.args[0].startswith("Atan has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.atanh([])
    assert (e_info.value.args[0].startswith("Atanh has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.ceil([])
    assert (e_info.value.args[0].startswith("Ceil has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.cos([])
    assert (e_info.value.args[0].startswith("Cos has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.cosh([])
    assert (e_info.value.args[0].startswith("Cosh has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.div([])
    assert (e_info.value.args[0].startswith("Div has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.elu([])
    assert (e_info.value.args[0].startswith("Elu has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.equal([])
    assert (e_info.value.args[0].startswith("Equal has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.exp([])
    assert (e_info.value.args[0].startswith("Exp has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.floor([])
    assert (e_info.value.args[0].startswith("Floor has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.greater([])
    assert (e_info.value.args[0].startswith("Greater has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.identity([])
    assert (e_info.value.args[0].startswith("Identity has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.less([])
    assert (e_info.value.args[0].startswith("Less has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.log([])
    assert (e_info.value.args[0].startswith("Log has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.max([])
    assert (e_info.value.args[0].startswith("Max has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.mean([])
    assert (e_info.value.args[0].startswith("Mean has no arguments"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.min([])
    assert (e_info.value.args[0].startswith("Min has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.mul([])
    assert (e_info.value.args[0].startswith("Mul has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.neg([])
    assert (e_info.value.args[0].startswith("Neg has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.logical_not([])
    assert (e_info.value.args[0].startswith("Not has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.logical_or([])
    assert (e_info.value.args[0].startswith("Or has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.pow([])
    assert (e_info.value.args[0].startswith("Pow has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.reciprocal([])
    assert (
        e_info.value.args[0].startswith("Reciprocal has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.relu([])
    assert (e_info.value.args[0].startswith("Relu has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.sigmoid([])
    assert (e_info.value.args[0].startswith("Sigmoid has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.sin([])
    assert (e_info.value.args[0].startswith("Sin has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.sinh([])
    assert (e_info.value.args[0].startswith("Sinh has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.softsign([])
    assert (e_info.value.args[0].startswith("Softsign has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.sqrt([])
    assert (e_info.value.args[0].startswith("Sqrt has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.sub([])
    assert (e_info.value.args[0].startswith("Sub has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.sum([])
    assert (e_info.value.args[0].startswith("Sum has no arguments"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.tan([])
    assert (e_info.value.args[0].startswith("Tan has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.tanh([])
    assert (e_info.value.args[0].startswith("Tanh has invalid number of"))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
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


def test_initialized_input_fp32():

    builder1 = poponnx.Builder()
    builder1.addInputTensor(poponnx.TensorInfo("FLOAT", [100, 100]))
    proto1 = builder1.getModelProto()
    assert (len(proto1) > 0)

    builder2 = poponnx.Builder()
    builder2.addInitializedInputTensor(np.ones([100, 100], np.float32))
    proto2 = builder2.getModelProto()

    assert (len(proto2) > 0)

    assert (len(proto2) > len(proto1))


def test_initialized_input_int32():

    builder1 = poponnx.Builder()
    builder1.addInputTensor(poponnx.TensorInfo("INT32", [100, 100]))
    proto1 = builder1.getModelProto()
    assert (len(proto1) > 0)

    builder2 = poponnx.Builder()
    builder2.addInitializedInputTensor(np.ones([100, 100], np.int32))
    proto2 = builder2.getModelProto()

    assert (len(proto2) > 0)

    assert (len(proto2) > len(proto1))


def test_initialized_input_fp16():

    builder1 = poponnx.Builder()
    builder1.addInputTensor(poponnx.TensorInfo("FLOAT16", [100, 100]))
    proto1 = builder1.getModelProto()
    assert (len(proto1) > 0)

    builder2 = poponnx.Builder()
    builder2.addInitializedInputTensor(np.ones([100, 100], np.float16))
    proto2 = builder2.getModelProto()

    assert (len(proto2) > 0)

    assert (len(proto2) > len(proto1))


def test_initialized_input_bool():

    builder1 = poponnx.Builder()
    builder1.addInputTensor(poponnx.TensorInfo("BOOL", [100, 100]))
    proto1 = builder1.getModelProto()
    assert (len(proto1) > 0)

    builder2 = poponnx.Builder()
    builder2.addInitializedInputTensor(np.ones([100, 100], np.bool))
    proto2 = builder2.getModelProto()

    assert (len(proto2) > 0)

    assert (len(proto2) > len(proto1))


def test_inout_tensor_info():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)

    assert (builder.getInputTensorIds() == [i1, i2])
    assert (builder.getOutputTensorIds() == [o])
    assert (builder.getTensorShape(i1) == [1, 2, 32, 32])
    assert (builder.getTensorShape(i2) == [4, 2, 3, 3])

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.getTensorShape("NotAnId")
    assert (
        e_info.value.args[0].find("is not an input or output tensor. Must be"))


def test_add_int_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = 100
    builder.addNodeAttribute("test", val, set(o))
    res = builder.getInt64NodeAttribute("test", set(o))
    assert (res == val)


def test_add_int_vector_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = [100, 200, -1]
    builder.addNodeAttribute("test", val, set(o))
    res = builder.getInt64VectorNodeAttribute("test", set(o))
    assert (res == val)


def test_add_float_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = .1
    builder.addNodeAttribute("test", val, set(o))
    res = builder.getFloatNodeAttribute("test", set(o))
    assert (res == pytest.approx(val))


def test_add_float_vector_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = [100., -.1, 100.0, 10.0]
    builder.addNodeAttribute("test", val, set(o))
    res = builder.getFloatVectorNodeAttribute("test", set(o))
    assert (res == pytest.approx(val))


def test_add_string_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = "test"
    builder.addNodeAttribute("test", val, set(o))
    res = builder.getStringNodeAttribute("test", set(o))
    assert (res == val)


def test_add_string_vector_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = ["test", "test2", "test"]
    builder.addNodeAttribute("test", val, set(o))
    res = builder.getStringVectorNodeAttribute("test", set(o))
    assert (res == val)


def test_add_attribute_missing_node():

    builder = poponnx.Builder()
    val = 100
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.addNodeAttribute("test", val, set(("i", "j")))
        assert (e_info.value.args[0].find(
            "Could not find a node with outputs i, j."))


def test_has_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = 100
    builder.addNodeAttribute("test", val, set(o))
    res = builder.nodeHasAttribute("test", set(o))
    assert (res)
    res = builder.nodeHasAttribute("test2", set(o))
    assert (not res)


def test_get_all_attribute_names():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set(o))
    builder.addNodeAttribute("test2", -1, set(o))
    builder.addNodeAttribute("test3", "abba", set(o))
    res = builder.getAllNodeAttributeNames(set(o))
    assert (set(res) == set(("test", "test2", "test3")))

    res = builder.getFloatNodeAttribute("test", set(o))
    assert (res == pytest.approx(100.))

    res = builder.getInt64NodeAttribute("test2", set(o))
    assert (res == -1)

    res = builder.getStringNodeAttribute("test3", set(o))
    assert (res == "abba")


def test_get_conv_strides_attribute():

    strides = [1, 1]

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], strides, [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    res = builder.getInt64VectorNodeAttribute("strides", set(o))
    assert (res == strides)


def test_dont_override_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set(o))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.addNodeAttribute("test", 100, set(o))
        assert (e_info.value.args[0].find("Node already has attribute test."))


def test_get_attribute_doesnt_exist():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set(o))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.getFloatNodeAttribute("test1", set(o))
        assert (
            e_info.value.args[0].find("Node does not have an attribute test1"))


def test_remove_attribute():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = 100
    builder.addNodeAttribute("test", val, set(o))
    res = builder.nodeHasAttribute("test", set(o))
    assert (res)
    builder.removeNodeAttribute("test", set(o))
    res = builder.nodeHasAttribute("test", set(o))
    assert (not res)


def test_remove_attribute_doesnt_exist():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [4, 2, 3, 3]))

    o = builder.convolution([i1, i2], [1, 1], [0, 0, 0, 0], [1, 1], 1)

    builder.addOutputTensor(o)
    # Set then get
    val = 100
    builder.addNodeAttribute("test", val, set(o))
    res = builder.nodeHasAttribute("test", set(o))
    assert (res)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.removeNodeAttribute("test1", set(o))
        assert (e_info.value.args[0].find(
            "Cannot remove attribute test1 as it does not exist."))


def test_get_attribute_wrong_type_int():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set(o))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.getInt64NodeAttribute("test", set(o))
        assert (e_info.value.args[0].find("Node test is not an integer."))


def test_get_attribute_wrong_type_int_vector():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100., set(o))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.getInt64VectorNodeAttribute("test", set(o))
        assert (
            e_info.value.args[0].find("Node test is not an integer vector."))


def test_get_attribute_wrong_type_float():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100, set(o))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.getFloatNodeAttribute("test", set(o))
        assert (e_info.value.args[0].find("Node test is not a float."))


def test_get_attribute_wrong_type_float_vector():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100, set(o))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.getFloatVectorNodeAttribute("test", set(o))
        assert (e_info.value.args[0].find("Node test is not a float vector."))


def test_get_attribute_wrong_type_string():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100, set(o))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.getStringNodeAttribute("test", set(o))
        assert (e_info.value.args[0].find("Node test is not a string."))


def test_get_attribute_wrong_type_string_vector():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    o = builder.add((i1, i1))

    builder.addOutputTensor(o)
    builder.addNodeAttribute("test", 100, set(o))

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        builder.getStringVectorNodeAttribute("test", set(o))
        assert (e_info.value.args[0].find("Node test is not a string vector."))


def test_load_onnx_model_from_other_builder(tmpdir):

    # Run the first builder
    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, 1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.Session(
        fnModel=proto, dataFeed=dataFlow, outputdir=str(tmpdir))

    session.setDevice(getDevice())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1: np.array([1, 2], dtype=np.float32),
        i2: np.array([3, 4], dtype=np.float32)
    }
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)
    assert (np.array_equal(anchors[o], [4, 6]))

    # Run a builder that imports the model of the other builder and check the
    # output is still the same
    builder2 = poponnx.Builder(proto)
    translation = builder2.getTensorTranslation()

    dataFlow = poponnx.DataFlow(
        1, 1, {translation[o]: poponnx.AnchorReturnType("ALL")})
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(translation[o], "l1LossVal", 0.1)]

    proto2 = builder.getModelProto()
    session = poponnx.Session(
        fnModel=proto2,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir))

    session.setDevice(getDevice())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        translation[i1]: np.array([1, 2], dtype=np.float32),
        translation[i2]: np.array([3, 4], dtype=np.float32)
    }
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.array_equal(anchors[translation[o]], [4, 6]))


def test_load_onnx_model_from_file(tmpdir):

    # Create a builder, store the model in a file and load it into a different
    # builder.
    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)
    filename = tmpdir + "/model.onnx"
    with open(filename, 'wb') as out:
        out.write(builder.getModelProto())

    builder2 = poponnx.Builder(str(filename))
    translation = builder2.getTensorTranslation()

    dataFlow = poponnx.DataFlow(
        1, 1, {translation[o]: poponnx.AnchorReturnType("ALL")})
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(translation[o], "l1LossVal", 0.1)]

    proto = builder2.getModelProto()

    session = poponnx.Session(
        fnModel=proto,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir))

    session.setDevice(getDevice())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        translation[i1]: np.array([1, 2], dtype=np.float32),
        translation[i2]: np.array([3, 4], dtype=np.float32)
    }
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.array_equal(anchors[translation[o]], [4, 6]))
