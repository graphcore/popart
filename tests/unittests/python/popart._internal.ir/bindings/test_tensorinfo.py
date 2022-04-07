# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir
import pytest


def test_data_type_creation():
    """ Test that we can create a popart._internal.ir.TensorData enum. """
    _ir.DataType.UINT8
    _ir.DataType.INT8
    _ir.DataType.UINT16
    _ir.DataType.INT16
    _ir.DataType.INT32
    _ir.DataType.INT64
    _ir.DataType.UINT32
    _ir.DataType.UINT64
    _ir.DataType.BOOL
    _ir.DataType.FLOAT
    _ir.DataType.FLOAT16
    _ir.DataType.BFLOAT16
    _ir.DataType.DOUBLE
    _ir.DataType.COMPLEX64
    _ir.DataType.COMPLEX128
    _ir.DataType.STRING
    _ir.DataType.UNDEFINED


def test_tensor_info_construction():
    """ Test that we can construct a popart._internal.ir.TensorInfo object. """
    dType = _ir.DataType.FLOAT16
    dTypeStr = "FLOAT16"
    shape = [5, 5]
    shapeStr = "(5,5)"
    metaShape = [25, 1]
    _ir.TensorInfo(dType, shape)
    _ir.TensorInfo(dType, shape, metaShape)
    _ir.TensorInfo(dTypeStr, shapeStr)
    _ir.TensorInfo(dTypeStr, shape)


def test_tensor_info_set0():
    """ Test the set() method of a popart._internal.ir.TensorInfo object. """
    dTypeOld = _ir.DataType.FLOAT
    dTypeNew = _ir.DataType.INT8
    tensorInfo = _ir.TensorInfo(dTypeOld, [4, 2])
    tensorInfo.set(dTypeNew)
    assert tensorInfo.dataType() == dTypeNew


@pytest.mark.parametrize("dTypeOld,dTypeNew", [
    (_ir.DataType.FLOAT, _ir.DataType.INT8),
    (_ir.DataType.INT8, _ir.DataType.FLOAT),
])
@pytest.mark.parametrize("shapeOld,shapeNew", [
    ([1, 2], [3, 4]),
    ([3, 4], [5, 6]),
])
def test_tensor_info_set1(dTypeOld, dTypeNew, shapeOld, shapeNew):
    """ Test the set() method of a popart._internal.ir.TensorInfo object. """
    tensorInfo = _ir.TensorInfo(dTypeOld, shapeOld)
    tensorInfo.set(dTypeNew, shapeNew)
    assert tensorInfo.shape() == shapeNew
    assert tensorInfo.dataType() == dTypeNew


@pytest.mark.parametrize("dTypeOld,dTypeNew", [
    (_ir.DataType.FLOAT, _ir.DataType.INT8),
    (_ir.DataType.INT8, _ir.DataType.FLOAT),
])
@pytest.mark.parametrize("shapeOld,shapeNew", [
    ([1, 2], [3, 4]),
    ([3, 4], [5, 6]),
])
@pytest.mark.parametrize("metaShapeOld,metaShapeNew", [
    ([1, 2], [3, 4]),
    ([3, 4], [5, 6]),
])
def test_tensor_info_set2(dTypeOld, dTypeNew, shapeOld, shapeNew, metaShapeOld,
                          metaShapeNew):
    """ Test the set() method of a popart._internal.ir.TensorInfo object. """
    tensorInfo = _ir.TensorInfo(dTypeOld, shapeOld, metaShapeOld)
    tensorInfo.set(dTypeNew, shapeNew, metaShapeNew)
    assert tensorInfo.shape() == shapeNew
    assert tensorInfo.metaShape() == metaShapeNew
    assert tensorInfo.dataType() == dTypeNew


@pytest.mark.parametrize("dType,shape,expectedShape", [
    (_ir.DataType.FLOAT, [2, 3], [2, 3]),
    ("FLOAT", "(4,5)", [4, 5]),
])
def test_tensor_info_shape(dType, shape, expectedShape):
    """ Test the shape() method of a popart._internal.ir.TensorInfo object. """
    tensorInfo = _ir.TensorInfo(dType, shape)
    assert tensorInfo.shape() == expectedShape


@pytest.mark.parametrize("metaShape", [[2, 3], [1, 6]])
def test_tensor_info_meta_shape(metaShape):
    """ Test the metaShape() method of a popart._internal.ir.TensorInfo object.
    """
    tensorInfo = _ir.TensorInfo(_ir.DataType.FLOAT, [6], metaShape)
    assert tensorInfo.metaShape() == metaShape


@pytest.mark.parametrize("shape", [[1], [2, 2], [3, 3, 3]])
def test_tensor_info_rank(shape):
    """ Test the rank() method of a popart._internal.ir.TensorInfo object. """
    tensorInfo = _ir.TensorInfo(_ir.DataType.FLOAT, shape)
    assert tensorInfo.rank() == len(shape)


@pytest.mark.parametrize("shape", [[1], [2, 2], [3, 3, 3]])
def test_tensor_info_nelms(shape):
    """ Test the nelms() method of a popart._internal.ir.TensorInfo object. """
    tensorInfo = _ir.TensorInfo(_ir.DataType.FLOAT, shape)
    assert tensorInfo.nelms() == np.prod(shape)


@pytest.mark.parametrize("dType,shape,nBytesInDType", [
    (_ir.DataType.FLOAT, [1], 4),
    (_ir.DataType.FLOAT16, [2, 2], 2),
    (_ir.DataType.INT8, [3, 3, 3], 1),
])
def test_tensor_info_nelms(dType, shape, nBytesInDType):
    """ Test the nelms() method of a popart._internal.ir.TensorInfo object. """
    tensorInfo = _ir.TensorInfo(dType, shape)
    assert tensorInfo.nbytes() == np.prod(shape) * nBytesInDType


@pytest.mark.parametrize("shape", [[1], [2, 2], [3, 3, 3]])
def test_tensor_info_dim(shape):
    """ Test the dim() method of a popart._internal.ir.TensorInfo object. """
    tensorInfo = _ir.TensorInfo(_ir.DataType.FLOAT, shape)
    for i, dim in enumerate(shape):
        assert tensorInfo.dim(i) == dim


@pytest.mark.parametrize("dType", [_ir.DataType.FLOAT, _ir.DataType.INT8])
def test_tensor_info_data_type(dType):
    """ Test the dataType() method of a popart._internal.ir.TensorInfo object.
    """
    tensorInfo = _ir.TensorInfo(dType, [6])
    assert tensorInfo.dataType() == dType


@pytest.mark.parametrize("dType,dTypeStr", [
    (_ir.DataType.FLOAT16, "FLOAT16"),
    ("INT8", "INT8"),
])
def test_tensor_info_data_type(dType, dTypeStr):
    """ Test the data_type() method of a popart._internal.ir.TensorInfo object.
    """
    tensorInfo = _ir.TensorInfo(dType, [5, 5])
    assert tensorInfo.data_type() == dTypeStr


@pytest.mark.parametrize("dType,dTypeLowerCase", [
    (_ir.DataType.FLOAT16, "float16"),
    ("INT8", "int8"),
])
def test_tensor_info_data_type_lcase(dType, dTypeLowerCase):
    """ Test the data_type_lcase() method of a popart._internal.ir.TensorInfo
    object.
    """
    tensorInfo = _ir.TensorInfo(dType, [5, 5])
    assert tensorInfo.data_type_lcase() == dTypeLowerCase


@pytest.mark.parametrize("shape", [[2, 4, 1, 3], [5], [1], [], [1000, 2]])
@pytest.mark.parametrize("np_dtype,ir_dtype", [
    (np.float32, _ir.DataType.FLOAT),
    (np.float16, _ir.DataType.FLOAT16),
    (np.int32, _ir.DataType.INT32),
    (np.int16, _ir.DataType.INT16),
    (np.uint32, _ir.DataType.UINT32),
    (np.bool_, _ir.DataType.BOOL),
])
def test_tensor_strides(shape, np_dtype, ir_dtype):
    """Test the strides function. See
     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html

    Args:
        shape (list[int]): Shape to use
        np_dtype (np.dtype): Numpy datatype
        ir_dtype (_ir.DataType): Corresponding PopART datatype
    """
    tInfo = _ir.TensorInfo(ir_dtype, shape)
    arr = np.zeros(shape=shape, dtype=np_dtype)
    assert tuple(tInfo.strides(shape)) == arr.strides


@pytest.mark.parametrize("dType1", [_ir.DataType.FLOAT16, _ir.DataType.FLOAT])
@pytest.mark.parametrize("shape1, metaShape1", [([2, 3], [6]), ([4, 5], [20])])
@pytest.mark.parametrize("dType2", [_ir.DataType.FLOAT16, _ir.DataType.FLOAT])
@pytest.mark.parametrize("shape2, metaShape2", [([2, 3], [6]), ([4, 5], [20])])
def test_tensor_info_data_operator_eq_neq(dType1, shape1, metaShape1, dType2,
                                          shape2, metaShape2):
    """ Test the operator==() and operator!=() methods of a
    popart._internal.ir.TensorInfo object.
    """
    tensorInfo1 = _ir.TensorInfo(dType1, shape1, metaShape1)
    tensorInfo2 = _ir.TensorInfo(dType2, shape2, metaShape2)
    areEqual = all(
        [dType1 == dType2, shape1 == shape2, metaShape1 == metaShape2])
    assert (tensorInfo1 == tensorInfo2) == areEqual
