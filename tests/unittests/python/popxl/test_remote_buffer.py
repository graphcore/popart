# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Tuple, Iterator
import numpy as np
import pytest
import popxl
from popxl.dtypes import dtype
from popxl.dtypes import int8, int16
from popxl.remote_buffer import RemoteBuffer, remote_buffer


@pytest.fixture(scope="function")
def input() -> Iterator[Tuple[Tuple[int, ...], dtype, int]]:
    """Return standard input to the remote buffer handle for the test.

    Yields:
        Iterator[Tuple[Tuple[int, ...], dtype, int]]: The standard buffer handle input
    """
    t_shape = (0, )
    t_dtype = int8
    entries = 1
    yield t_shape, t_dtype, entries


@pytest.fixture(scope="function")
def standard_remote_buffer(
        input: Tuple[Tuple[int, ...], dtype, int]) -> Iterator[RemoteBuffer]:
    """Return the standard remote buffer handle for the tests.

    Args:
        input (Tuple[Tuple[int, ...], dtype, int]): The standard remote buffer handle input

    Yields:
        Iterator[RemoteBuffer]: The standard remote buffer handle
    """
    t_shape, t_dtype, entries = input
    standard_remote_buffer = RemoteBuffer(ir=popxl.Ir(),
                                          tensor_shape=t_shape,
                                          tensor_dtype=t_dtype,
                                          entries=entries)
    yield standard_remote_buffer


class TestRemoteBuffer:
    def test___init__(self, input: Tuple[Tuple[int, ...], dtype, int]) -> None:
        """Test that the remote buffer handles are correctly initialized.

        Args:
            input (Tuple[Tuple[int, ...], dtype, int]): The standard remote buffer handle input
        """
        # Test the initializer
        t_shape, t_dtype, entries = input
        ir = popxl.Ir()
        standard_remote_buffer = RemoteBuffer(ir=ir,
                                              tensor_shape=t_shape,
                                              tensor_dtype=t_dtype,
                                              entries=entries)
        assert standard_remote_buffer.remote_buffer_id == 1
        assert standard_remote_buffer.tensor_shape == t_shape
        assert standard_remote_buffer.tensor_dtype == t_dtype
        assert standard_remote_buffer.entries == entries

        # Test that the buffer id is incremented
        remote_buffer_2 = RemoteBuffer(ir=ir,
                                       tensor_shape=t_shape,
                                       tensor_dtype=t_dtype,
                                       entries=entries)
        assert remote_buffer_2.remote_buffer_id == 2

        # Assert that entries have to be positive
        with pytest.raises(ValueError) as e_info:
            _ = RemoteBuffer(ir=popxl.Ir(),
                             tensor_shape=t_shape,
                             tensor_dtype=t_dtype,
                             entries=0)
        assert e_info.value.args[0].startswith(
            "Entries must be a non-zero, positive integer")

    def test_tensor_shape(self, standard_remote_buffer: RemoteBuffer) -> None:
        """Test the setters on tensor_shape works as expected.

        Args:
            standard_remote_buffer (RemoteBuffer): The standard remote buffer handle used for
              testing.
        """
        # Test that it's not possible to reset the shape
        standard_remote_buffer = standard_remote_buffer
        with pytest.raises(AttributeError) as e_info:
            standard_remote_buffer.tensor_shape = (
                11, 11, 11)  # This is not ok since it's a new value
        assert e_info.value.args[0] == "can't set attribute"

    def test_tensor_dtype(self, standard_remote_buffer: RemoteBuffer) -> None:
        """Test the setters on tensor_dtype works as expected.

        Args:
            standard_remote_buffer (RemoteBuffer): The standard remote buffer handle used for
              testing.
        """
        # Test that it's not possible to reset the data type
        standard_remote_buffer = standard_remote_buffer
        with pytest.raises(AttributeError) as e_info:
            standard_remote_buffer.tensor_dtype = int16  # This is not ok since it's a new value
        assert e_info.value.args[0] == "can't set attribute"

    def test_entries(self, input: Tuple[Tuple[int, ...], dtype, int],
                     standard_remote_buffer: RemoteBuffer) -> None:
        """Test the setters on entires works as expected.

        Args:
            input (Tuple[Tuple[int, ...], dtype, int]): The standard remote buffer handle input
            standard_remote_buffer (RemoteBuffer): The standard remote buffer handle used for
              testing.
        """
        # Test that it's possible to reset entries
        standard_remote_buffer = standard_remote_buffer
        standard_remote_buffer.entries = 2
        assert standard_remote_buffer.entries == 2

        # Check that non-positive values are not allowed
        with pytest.raises(ValueError) as e_info:
            t_shape, t_dtype, _ = input
            _ = RemoteBuffer(ir=popxl.Ir(),
                             tensor_shape=t_shape,
                             tensor_dtype=t_dtype,
                             entries=0)
        assert e_info.value.args[0].startswith(
            "Entries must be a non-zero, positive integer")
        with pytest.raises(ValueError) as e_info:
            standard_remote_buffer.entries = 0
        assert e_info.value.args[0].startswith(
            "Entries must be a non-zero, positive integer")

    def test_set_remote_buffer_info(
            self, input: Tuple[Tuple[int, ...], dtype, int],
            standard_remote_buffer: RemoteBuffer) -> None:
        """Test that once can set, but not reset the remote buffer info.

        Args:
            input (Tuple[Tuple[int, ...], dtype, int]): The standard remote buffer handle input
            standard_remote_buffer (RemoteBuffer): The standard remote buffer handle used for
              testing.
        """
        t_shape, t_dtype, entries = input

        # Make tensor to check against
        ir = popxl.Ir()
        main = ir.main_graph
        with main:
            t = popxl.variable(data=np.zeros(t_shape), dtype=t_dtype)

        # Create the handles
        remote_buffer_1 = standard_remote_buffer
        remote_buffer_1.validate_tensor_matches_buffer(t)
        # Create an instance equivalent of standard_remote_buffer
        remote_buffer_2 = RemoteBuffer(ir=popxl.Ir(),
                                       tensor_shape=t_shape,
                                       tensor_dtype=t_dtype,
                                       entries=entries)
        remote_buffer_2.validate_tensor_matches_buffer(t)
        # Create an instance equivalent of standard_remote_buffer, but with shape changed
        remote_buffer_3 = RemoteBuffer(ir=popxl.Ir(),
                                       tensor_shape=(1, 3, 5),
                                       tensor_dtype=t_dtype,
                                       entries=entries)
        # Create an instance equivalent of standard_remote_buffer, but with dtype changed
        remote_buffer_4 = RemoteBuffer(ir=popxl.Ir(),
                                       tensor_shape=t_shape,
                                       tensor_dtype=int16,
                                       entries=entries)

        assume_fails = (remote_buffer_3, remote_buffer_4)
        for assume_fail in assume_fails:
            with pytest.raises(ValueError) as e_info:
                assume_fail.validate_tensor_matches_buffer(t)
            assert e_info.value.args[0].startswith(
                "Tensor does not match buffer.")

    def test_two_irs(self, input: Tuple[Tuple[int, ...], dtype, int]) -> None:
        """Test that the buffers of two IRs act independently.

        Args:
            input (Tuple[Tuple[int, ...], dtype, int]): The standard remote buffer handle input
        """
        t_shape, t_dtype, entries = input

        ir_1 = popxl.Ir()
        ir_2 = popxl.Ir()

        rb_1 = RemoteBuffer(ir=ir_1,
                            tensor_shape=t_shape,
                            tensor_dtype=t_dtype,
                            entries=entries)

        rb_2 = RemoteBuffer(ir=ir_2,
                            tensor_shape=t_shape,
                            tensor_dtype=t_dtype,
                            entries=entries)

        assert rb_1.remote_buffer_id == rb_2.remote_buffer_id

    def test_set_remote_buffer_info(
            self, standard_remote_buffer: RemoteBuffer) -> None:
        """Test that it's possible to reset the remote buffer.

        Args:
            standard_remote_buffer (RemoteBuffer): The standard remote buffer handle used for
              testing.
        """
        t_shape = (3, 11, 5)
        t_dtype = int16
        entries = 42
        standard_remote_buffer.set_remote_buffer_info(tensor_dtype=t_dtype,
                                                      tensor_shape=t_shape,
                                                      entries=entries)

        assert standard_remote_buffer.tensor_shape == t_shape
        assert standard_remote_buffer.tensor_dtype == t_dtype
        assert standard_remote_buffer.entries == entries

    def test_reset_entries(self, input: Tuple[Tuple[int, ...], dtype, int],
                           standard_remote_buffer: RemoteBuffer) -> None:
        """Test that it's possible to reset the entries.

        Args:
            input (Tuple[Tuple[int, ...], dtype, int]): The standard remote buffer handle input
            standard_remote_buffer (RemoteBuffer): The standard remote buffer handle used for
              testing.
        """
        t_shape, t_dtype, entries = input
        entries += 1
        standard_remote_buffer.entries = entries

        assert standard_remote_buffer.tensor_shape == t_shape
        assert standard_remote_buffer.tensor_dtype == t_dtype
        assert standard_remote_buffer.entries == entries

    def test_remote_buffer(self,
                           input: Tuple[Tuple[int, ...], dtype, int]) -> None:
        """Test that creating the a remote buffer from the context works.

        Args:
            input (Tuple[Tuple[int, ...], dtype, int]): The standard remote buffer handle input
        """
        ir = popxl.Ir()
        # Creating without context
        t_shape, t_dtype, entries = input
        remote_buffer_class = RemoteBuffer(ir=ir,
                                           tensor_shape=t_shape,
                                           tensor_dtype=t_dtype,
                                           entries=entries)
        # Create with context
        main = ir.main_graph

        with main:
            remote_buffer_context = remote_buffer(tensor_shape=t_shape,
                                                  tensor_dtype=t_dtype,
                                                  entries=entries)

        assert remote_buffer_class._current_ir == remote_buffer_context._current_ir
        assert remote_buffer_class.tensor_shape == remote_buffer_context.tensor_shape
        assert remote_buffer_class.tensor_dtype == remote_buffer_context.tensor_dtype
        assert remote_buffer_class.entries == remote_buffer_context.entries
