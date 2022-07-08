# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart._internal.ir as _ir


def test_tensor_data_construction():
    """Test that we can construct a popart._internal.ir.TensorData object."""
    a = np.random.rand(2, 3, 4)
    tInfo = _ir.TensorInfo(_ir.DataType.FLOAT, a.shape)
    _ = _ir.TensorData(tInfo, a)


def test_tensor_data_reset_data():
    """Test the resetData() method of a popart._internal.ir.TensorData object."""
    a = np.random.rand(2, 3, 4)
    b = np.random.rand(2, 3, 4)
    tInfo = _ir.TensorInfo(_ir.DataType.FLOAT, a.shape)
    tData = _ir.TensorData(tInfo, a)
    tData.resetData(tInfo, b)
