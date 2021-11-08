# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


class TestRemoteStore:
    @pytest.mark.parametrize("use_offset", [True, False])
    @pytest.mark.parametrize("use_remote_buffer_id", [True, False])
    def test_only_t_as_input(self, use_offset: bool,
                             use_remote_buffer_id: bool) -> None:
        """Test that the graph is correct when using the remote store op

        Args:
            use_offset (bool): Whether or not to use offset
            use_remote_buffer_id (bool): Whether or not to set the remote buffer_id
        """
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            t = pir.variable([3, 5, 7])
            if use_offset:
                offset = pir.constant([1], name='offset')
                # With this option the graph should contain
                # 1. t
                # 2. offset
                n_tensors = 2
            else:
                offset = None
                # With this option the graph should contain
                # 1. t
                n_tensors = 1

            remote_buffer_id = 1 if use_remote_buffer_id else -1

            if remote_buffer_id == -1:
                with pytest.raises(NotImplementedError):
                    ops.remote_store(t, offset, remote_buffer_id)
                return

            ops.remote_store(t, offset, remote_buffer_id)

        assert len(g.get_tensors()) == n_tensors
        # Only t is a variable
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("RemoteStore",
                                   _ir.op.exchange.RemoteStoreOp, g)
