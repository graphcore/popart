# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
from popart.ir.dtypes import int8, int16
from popart.ir.remote_buffer_handle import RemoteBufferHandle


class TestRemoteBufferHandle:
    def test___new___(self):
        """Test that the remote buffer handles are correctly created."""
        rbh_default_1 = RemoteBufferHandle()
        rbh_default_2 = RemoteBufferHandle()

        # The second call to RemoteBufferHandle() should increase the remote_buffer_id
        assert rbh_default_1.remote_buffer_id == 1
        assert rbh_default_2.remote_buffer_id == 2
        assert rbh_default_1 != rbh_default_2

        # Test that no new instance is created
        rbh_3_1 = RemoteBufferHandle(remote_buffer_id=3)
        rbh_3_2 = RemoteBufferHandle(remote_buffer_id=3)
        assert rbh_3_1 == rbh_3_2

        # Clean-up so that the RemoteBufferHandle gets reset
        RemoteBufferHandle._buffers = {}

    def test___init__(self):
        """Test that the remote buffer handles are correctly initialized."""
        # Test that illegal values are captured
        with pytest.raises(NotImplementedError) as e_info:
            _ = RemoteBufferHandle(remote_buffer_id=-1)
            assert e_info.value.args[0] == (
                "remote_buffer_id = -1 (automatic RemoteSetup) "
                "not supported")

        # Clean-up so that the RemoteBufferHandle gets reset
        RemoteBufferHandle._buffers = {}

    def test_tensor_shape(self):
        """Test the setters on tensor_shape works as expected."""
        # Test that attributes are set to all variables of the same instance
        rbh_default_1_1 = RemoteBufferHandle()
        rbh_default_1_2 = RemoteBufferHandle(remote_buffer_id=1)
        assert rbh_default_1_1 == rbh_default_1_2
        rbh_default_1_1.tensor_shape = (1, 3, 5)
        assert rbh_default_1_1.tensor_shape == rbh_default_1_2.tensor_shape
        # Check that shape cannot be written to twice
        rbh_default_1_1.tensor_shape = (
            1, 3, 5)  # This is ok since it's the same value
        rbh_default_1_2.tensor_shape = (
            1, 3, 5)  # This is ok since it's the same value
        with pytest.raises(ValueError) as e_info:
            rbh_default_1_1.tensor_shape = (
                11, 11, 11)  # This is not ok since it's a new value
        assert e_info.value.args[0].startswith("Cannot reset buffer shape")
        with pytest.raises(ValueError) as e_info:
            rbh_default_1_2.tensor_shape = (
                11, 11, 11)  # This is not ok since it's a new value
        assert e_info.value.args[0].startswith("Cannot reset buffer shape")

        # Test the same when shape is set in the constructor
        rbh_default_2_1 = RemoteBufferHandle(remote_buffer_id=2,
                                             tensor_shape=(1, 3, 5))
        rbh_default_2_2 = RemoteBufferHandle(remote_buffer_id=2,
                                             tensor_shape=(1, 3, 5))
        assert rbh_default_2_1 == rbh_default_2_2
        assert rbh_default_2_1.tensor_shape == rbh_default_2_2.tensor_shape
        # Check that shape cannot be written to twice
        rbh_default_2_1.tensor_shape = (
            1, 3, 5)  # This is ok since it's the same value
        rbh_default_2_2.tensor_shape = (
            1, 3, 5)  # This is ok since it's the same value
        with pytest.raises(ValueError) as e_info:
            rbh_default_2_1.tensor_shape = (
                11, 11, 11)  # This is not ok since it's a new value
        assert e_info.value.args[0].startswith("Cannot reset buffer shape")
        with pytest.raises(ValueError) as e_info:
            rbh_default_2_2.tensor_shape = (
                11, 11, 11)  # This is not ok since it's a new value
        assert e_info.value.args[0].startswith("Cannot reset buffer shape")
        # It should not be possible to edit the shape from the constructor
        with pytest.raises(ValueError) as e_info:
            _ = RemoteBufferHandle(remote_buffer_id=2,
                                   tensor_shape=(11, 11, 11))
        assert e_info.value.args[0].startswith("Cannot reset buffer shape")

        # Clean-up so that the RemoteBufferHandle gets reset
        RemoteBufferHandle._buffers = {}

    def test_tensor_dtype(self):
        """Test the setters on tensor_dtype works as expected."""
        # Test that attributes are set to all variables of the same instance
        rbh_default_1_1 = RemoteBufferHandle()
        rbh_default_1_2 = RemoteBufferHandle(remote_buffer_id=1)
        assert rbh_default_1_1 == rbh_default_1_2
        rbh_default_1_1.tensor_dtype = int8
        assert rbh_default_1_1.tensor_dtype == rbh_default_1_2.tensor_dtype
        # Check that dtype cannot be written to twice
        rbh_default_1_1.tensor_dtype = int8  # This is ok since it's the same value
        rbh_default_1_2.tensor_dtype = int8  # This is ok since it's the same value
        with pytest.raises(ValueError) as e_info:
            rbh_default_1_1.tensor_dtype = int16  # This is not ok since it's a new value
        assert e_info.value.args[0].startswith("Cannot reset buffer dtype")
        with pytest.raises(ValueError) as e_info:
            rbh_default_1_2.tensor_dtype = int16  # This is not ok since it's a new value
        assert e_info.value.args[0].startswith("Cannot reset buffer dtype")

        # Test the same when dtype is set in the constructor
        rbh_default_2_1 = RemoteBufferHandle(remote_buffer_id=2,
                                             tensor_dtype=int8)
        rbh_default_2_2 = RemoteBufferHandle(remote_buffer_id=2,
                                             tensor_dtype=int8)
        assert rbh_default_2_1 == rbh_default_2_2
        assert rbh_default_2_1.tensor_dtype == rbh_default_2_2.tensor_dtype
        # Check that dtype cannot be written to twice
        rbh_default_2_1.tensor_dtype = int8  # This is ok since it's the same value
        rbh_default_2_2.tensor_dtype = int8  # This is ok since it's the same value
        with pytest.raises(ValueError) as e_info:
            rbh_default_2_1.tensor_dtype = int16  # This is not ok since it's a new value
        assert e_info.value.args[0].startswith("Cannot reset buffer dtype")
        with pytest.raises(ValueError) as e_info:
            rbh_default_2_2.tensor_dtype = int16  # This is not ok since it's a new value
        assert e_info.value.args[0].startswith("Cannot reset buffer dtype")
        # It should not be possible to edit the dtype from the constructor
        with pytest.raises(ValueError) as e_info:
            _ = RemoteBufferHandle(remote_buffer_id=2, tensor_dtype=int16)
        assert e_info.value.args[0].startswith("Cannot reset buffer dtype")

        # Clean-up so that the RemoteBufferHandle gets reset
        RemoteBufferHandle._buffers = {}

    def test_repeats(self):
        """Test the setters on repeats works as expected."""
        # Test that attributes are set to all variables of the same instance
        rbh_default_1_1 = RemoteBufferHandle()
        rbh_default_1_2 = RemoteBufferHandle(remote_buffer_id=1)
        assert rbh_default_1_1 == rbh_default_1_2
        rbh_default_1_1.repeats = 1
        assert rbh_default_1_1.repeats == rbh_default_1_2.repeats
        # Check that repeats changes across the variables
        rbh_default_1_1.repeats = 2
        assert rbh_default_1_1.repeats == rbh_default_1_2.repeats
        rbh_default_1_2.repeats = 3
        assert rbh_default_1_1.repeats == rbh_default_1_2.repeats

        # Check that it can be set from the constructor
        rbh_default_1_3 = RemoteBufferHandle(remote_buffer_id=1, repeats=4)
        assert rbh_default_1_1.repeats == rbh_default_1_2.repeats
        assert rbh_default_1_3.repeats == rbh_default_1_2.repeats

        # Check that non-positive values are not allowed
        with pytest.raises(ValueError) as e_info:
            _ = RemoteBufferHandle(remote_buffer_id=2, repeats=0)
        assert e_info.value.args[0].startswith(
            "Repeats must be a non-zero, positive integer")
        with pytest.raises(ValueError) as e_info:
            rbh_default_1_2.repeats = 0
        assert e_info.value.args[0].startswith(
            "Repeats must be a non-zero, positive integer")

        # Clean-up so that the RemoteBufferHandle gets reset
        RemoteBufferHandle._buffers = {}
