# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest

import popart.ir as pir
from popart.ir.globals import gcg


class TestGraph:
    def test_context_manager(self):
        ir = pir.Ir()
        main = ir.main_graph()

        exp_prefix = 'Trying to access a graph, but no graph has been selected.'

        with pytest.raises(RuntimeError) as excinfo:
            gcg()
        assert str(excinfo.value).startswith(exp_prefix)

        with main:
            assert gcg() == main

        with pytest.raises(RuntimeError) as excinfo:
            gcg()
        assert str(excinfo.value).startswith(exp_prefix)

    def test_tensor_id_conflict(self):
        ir = pir.Ir()
        main = ir.main_graph()
        with main:
            name0 = pir.variable(1, name="tensor").id
            name1 = pir.variable(1, name="tensor").id
            name2 = pir.constant(1, name="tensor").id
        assert name0 == "tensor"
        ids = [name0, name1, name2]
        assert len(ids) == len(set(ids))
