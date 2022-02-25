# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Tuple
import pytest
import numpy as np
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir.remote_buffer import RemoteBuffer
from popart.ir.dtypes import dtype, float16, float32
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestRemoteStore:
    @pytest.mark.parametrize("tensor_shape", [(7, 11, 13), (17, 19)])
    @pytest.mark.parametrize("entries", [3, 5])
    @pytest.mark.parametrize("tensor_dtype", [float16, float32])
    @pytest.mark.parametrize("offset_as_int", [True, False])
    def test_remote_store_graph(self, tensor_shape: Tuple[int, ...],
                                entries: int, tensor_dtype: dtype,
                                offset_as_int: bool) -> None:
        """Test that the graph is correct when using the remote store op.

        Args:
            tensor_shape (Tuple[int, ...]): The shape of the tensor to be stored
            entries (int): The number of tensors to potentially store in the buffer
            tensor_dtype (dtype): The type of the tensors to be stored
            offset_as_int (bool): If true the offset input to the op will be given as an int
        """
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            t = pir.variable(
                np.random.rand(*tensor_shape).astype(tensor_dtype.as_numpy()))

            if offset_as_int:
                offset = 0
            else:
                offset = pir.constant([0], name='offset')
            # With this option the graph should contain
            # 1. t
            # 2. offset
            n_tensors = 2

            remote_buffer = RemoteBuffer(ir=ir,
                                         tensor_shape=tensor_shape,
                                         tensor_dtype=tensor_dtype,
                                         entries=entries)

            ops.remote_store(remote_buffer, offset, t)

        assert len(g.tensors) == n_tensors
        # Only t is a variable
        assert len(g.variables) == 1
        assert contains_op_of_type("RemoteStore",
                                   _ir.op.exchange.RemoteStoreOp, g)
