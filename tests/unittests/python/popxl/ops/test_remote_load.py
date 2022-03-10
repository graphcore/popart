# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Tuple
import pytest
import numpy as np
import popxl
import popxl.ops as ops
from popxl.remote_buffer import RemoteBuffer
from popxl.dtypes import dtype, float16, float32
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestRemoteLoad:
    @pytest.mark.parametrize("tensor_shape", [(7, 11, 13), (17, 19)])
    @pytest.mark.parametrize("entries", [3, 5])
    @pytest.mark.parametrize("tensor_dtype", [float16, float32])
    @pytest.mark.parametrize("offset_as_int", [True, False])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_remote_load_graph(self, tensor_shape: Tuple[int, ...],
                               entries: int, tensor_dtype: dtype,
                               offset_as_int: bool, inplace: bool) -> None:
        """Test that the graph is correct when using the remote load op

        Args:
            tensor_shape (Tuple[int, ...]): The shape of the tensor to be loaded
            entries (int): The number of tensors potentially stored in the buffer
            tensor_dtype (dtype): The type of the tensors to be loaded
            offset_as_int (bool): If true the offset input to the op will be given as an int
            inplace (bool): Whether or not to use the inplace version of the op
        """
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            if offset_as_int:
                offset = 0
            else:
                offset = popxl.constant([0], name='offset')
            # With this option the graph should contain
            # 1. t
            # 2. offset
            # 3. out
            n_tensors = 3

            remote_buffer = RemoteBuffer(tensor_shape=tensor_shape,
                                         tensor_dtype=tensor_dtype,
                                         entries=entries)

            if not inplace:
                ops.remote_load(remote_buffer, offset)
                n_variables = 0
            else:
                t = popxl.variable(
                    np.random.rand(*tensor_shape).astype(
                        tensor_dtype.as_numpy()))
                ops.remote_load_(remote_buffer, offset, t)
                # t is the only variable
                n_variables = 1

        assert len(g.tensors) == n_tensors
        # Only t is a variable
        assert len(g.variables) == n_variables
        type_string = "RemoteLoad" if not inplace else "RemoteLoadInplace"
        pb_type = _ir.op.exchange.RemoteLoadOp if not inplace else _ir.op.exchange.RemoteLoadInplaceOp
        assert contains_op_of_type(type_string, pb_type, g)
