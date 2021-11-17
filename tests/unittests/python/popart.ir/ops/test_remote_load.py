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
    @pytest.mark.parametrize("use_remote_buffer_id", [True, False])
    @pytest.mark.parametrize("tensor_shape", [(7, 11, 13), (17, 19)])
    @pytest.mark.parametrize("repeats", [3, 5])
    @pytest.mark.parametrize("tensor_dtype", [float16, float32])
    def test_remote_load_graph(self, use_offset: bool,
                               use_remote_buffer_id: bool,
                               tensor_shape: Tuple[int, ...], repeats: int,
                               tensor_dtype: dtype) -> None:
        """Test that the graph is correct when using the remote load op

        Args:
            use_offset (bool): Whether or not to use offset
            use_remote_buffer_id (bool): Whether or not to set the remote buffer_id
            tensor_shape (Tuple[int, ...]): The shape of the tensor to be loaded
            repeats (int): The number of tensors potentially stored in the buffer
            tensor_dtype (dtype): The type of the tensors to be loaded
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

            remote_buffer_id = 1 if use_remote_buffer_id else -1

            if remote_buffer_id == -1:
                with pytest.raises(NotImplementedError):
                    _ = RemoteBufferHandle(remote_buffer_id=remote_buffer_id,
                                           tensor_shape=tensor_shape,
                                           tensor_dtype=tensor_dtype,
                                           repeats=repeats)
                # Clean-up so that the RemoteBufferHandle gets reset
                RemoteBufferHandle._buffers = {}
                return

            rbh = RemoteBufferHandle(remote_buffer_id=remote_buffer_id,
                                     tensor_shape=tensor_shape,
                                     tensor_dtype=tensor_dtype,
                                     repeats=repeats)
            ops.remote_load(t, offset, rbh)

        assert len(g.get_tensors()) == n_tensors
        # Only t is a variable
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("RemoteLoad", _ir.op.exchange.RemoteLoadOp,
                                   g)

        # Clean-up so that the RemoteBufferHandle gets reset
        RemoteBufferHandle._buffers = {}
