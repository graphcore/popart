# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
import popart.ir as pir
import popart.ir.ops as ops


class TestAdd:
    def contains_add(self, ir: pir.ir):
        ops = json.loads(ir._pb_ir.serializeToJSON())['maingraph']
        add_ops = list(filter(lambda op: op["type"] == "Add", ops))
        assert len(add_ops) == 1
        op = add_ops[0]
        assert op["type"] == "Add"
        assert op["version"] == "6"
        assert op["domain"] == "ai.onnx"

    def test_fn(self):
        ir = pir.Ir()
        with ir.main_graph():
            a = pir.variable(1)
            b = pir.variable(2)
            c = ops.add(a, b)
        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 2
        self.contains_add(ir)

    def test_dunder(self):
        ir = pir.Ir()
        with ir.main_graph():
            a = pir.variable(1)
            b = pir.variable(2)
            c = a + b
        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 2
        self.contains_add(ir)

    def test_ensure_tensor(self):
        ir = pir.Ir()
        with ir.main_graph():
            a = pir.variable(1)
            c = a + 2
        assert len(ir.main_graph().get_tensors()) == 3
        assert len(ir.main_graph().get_variables()) == 1
        assert len(ir.main_graph().get_constants()) == 1
        self.contains_add(ir)
