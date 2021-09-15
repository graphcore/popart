# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops


class TestAdd:
    def _contains_add(self, graph: pir.Graph):
        pb_g = graph._pb_graph

        for pb_op in pb_g.getOps():
            if pb_op.opType() == "Add":
                return True
        return False

    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = ops.add(a, b)

        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        self._contains_add(g)

    def test_dunder(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            a = pir.variable(1)
            b = pir.variable(2)
            c = a + b

        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        self._contains_add(g)

    def test_ensure_tensor(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with ir.main_graph():
            a = pir.variable(1)
            c = a + 2

        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 1
        assert len(g.get_constants()) == 1
        self._contains_add(g)
