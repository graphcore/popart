# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import pytest
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


def id_fn(x: popxl.Tensor):
    return x


def square_fn(x: popxl.Tensor):
    return x * x


@pytest.mark.parametrize("test_fn", [id_fn, square_fn])
class TestRemoteCodeLoad:
    def test_remote_code_load_fn(self, test_fn) -> None:
        """Test that the graph contains the correct op.
        """
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            id_graph = ir.create_graph(test_fn, a)
            ops.remote_code_load(id_graph, "executable")
            ops.call(id_graph, a)

        assert contains_op_of_type("RemoteCodeLoad",
                                   _ir.op.exchange.RemoteCodeLoadOp, g)

    def test_call_same_graph(self, test_fn) -> None:
        """Test that you cannot load the code for the same graph you are in the context for.
        """
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            a = popxl.variable(1)
            _ = ir.create_graph(test_fn, a)
            with pytest.raises(ValueError) as e_info:
                ops.remote_code_load(g, "executable")  # same as parent graph.
            assert e_info.value.args[0].startswith(
                f"The {ops.remote_code_load.__name__} op cannot load the code")
