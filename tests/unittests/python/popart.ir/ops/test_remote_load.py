# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Tuple
import pytest
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.remote_buffer_handle import RemoteBufferHandle
from popart.ir.dtypes import dtype, float16, float32
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestRemoteLoad:
    @pytest.mark.parametrize("use_offset", [True, False])
    @pytest.mark.parametrize("tensor_shape", [(7, 11, 13), (17, 19)])
    @pytest.mark.parametrize("repeats", [3, 5])
    @pytest.mark.parametrize("tensor_dtype", [float16, float32])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_remote_load_graph(self, use_offset: bool,
                               tensor_shape: Tuple[int, ...], repeats: int,
                               tensor_dtype: dtype, inplace: bool) -> None:
        """Test that the graph is correct when using the remote load op

        Args:
            use_offset (bool): Whether or not to use offset
            tensor_shape (Tuple[int, ...]): The shape of the tensor to be loaded
            repeats (int): The number of tensors potentially stored in the buffer
            tensor_dtype (dtype): The type of the tensors to be loaded
            inplace (bool): Whether or not to use the inplace version of the op
        """
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable(
                np.random.rand(*tensor_shape).astype(tensor_dtype.as_numpy()))
            if use_offset:
                offset = pir.constant([1], name='offset')
                # With this option the graph should contain
                # 1. t
                # 2. offset
                # 3. out
                n_tensors = 3
            else:
                offset = None
                # With this option the graph should contain
                # 1. t
                # 2. out
                n_tensors = 2

            rbh = RemoteBufferHandle(remote_buffer_id=1,
                                     tensor_shape=tensor_shape,
                                     tensor_dtype=tensor_dtype,
                                     repeats=repeats)

            op = ops.remote_load if not inplace else ops.remote_load_
            op(t, offset, rbh)

        assert len(g.get_tensors()) == n_tensors
        # Only t is a variable
        assert len(g.get_variables()) == 1
        type_string = "RemoteLoad" if not inplace else "RemoteLoadInplace"
        pb_type = _ir.op.exchange.RemoteLoadOp if not inplace else _ir.op.exchange.RemoteLoadInplaceOp
        assert contains_op_of_type(type_string, pb_type, g)

        # Clean-up so that the RemoteBufferHandle gets reset
        RemoteBufferHandle._buffers = {}

    def test_raises(self):
        """Test that remote_buffer_id=-1 raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _ = RemoteBufferHandle(remote_buffer_id=-1,
                                   tensor_shape=None,
                                   tensor_dtype=None,
                                   repeats=1)
        # Clean-up so that the RemoteBufferHandle gets reset
        RemoteBufferHandle._buffers = {}
