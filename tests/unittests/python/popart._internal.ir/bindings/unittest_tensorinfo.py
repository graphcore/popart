# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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


@pytest.mark.parametrize("dType,shape,expectedShape", [
    (_ir.DataType.FLOAT, [2, 3], [2, 3]),
    ("FLOAT", "(4,5)", [4, 5]),
])
def test_tensor_info_shape(dType, shape, expectedShape):
    """ Test the shape() method of a popart._internal.ir.TensorInfo object. """
    tensorInfo = _ir.TensorInfo(dType, shape)
    assert tensorInfo.shape() == expectedShape
