# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest


@pytest.mark.parametrize("condition", [True, False])
def test_if_basic(op_tester, condition):
    i1 = np.array([4]).astype(np.float32)

    def init_builder(builder):
        builder.setGraphName("main_graph")
        inp = builder.addInputTensor(i1)  # input
        cond = builder.aiOnnx.constant(np.array(condition).astype(bool), "cond")
        o1 = builder.aiOnnx.constant(np.array([1]).astype(np.float32), "o1")

        then_builder = builder.createSubgraphBuilder()
        then_builder.setGraphName("body_then")
        out_then = then_builder.aiOnnx.add([o1, inp])
        then_builder.addOutputTensor(out_then)

        else_builder = builder.createSubgraphBuilder()
        else_builder.setGraphName("body_else")
        o2 = else_builder.aiOnnx.constant(np.array([2]).astype(np.float32), "o2")
        out_else = else_builder.aiOnnx.add([inp, o2])
        else_builder.addOutputTensor(out_else)

        o = builder.aiOnnx.logical_if([cond], 1, else_builder, then_builder)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        return [
            (np.array([1.0]) if condition else np.array([2.0])).astype(np.float32) + i1
        ]

    op_tester.run(init_builder, reference, step_type="infer")


@pytest.mark.parametrize("condition", [True, False])
def test_if_const_implicit_tensor(op_tester, condition):
    i1 = np.array([4]).astype(np.int32)

    def init_builder(builder):
        builder.setGraphName("main_graph")
        inp = builder.addInputTensor(i1)  # input
        cond = builder.aiOnnx.constant(np.array(condition).astype(bool), "cond")
        k_in = builder.aiOnnx.constant(np.array([1]).astype(np.int64), "k")

        then_builder = builder.createSubgraphBuilder()
        then_builder.setGraphName("body_then")
        out_then = then_builder.aiOnnx.topk([inp, k_in], axis=0)[0]
        out_then = then_builder.aiOnnx.add([out_then, inp])
        then_builder.addOutputTensor(out_then)

        else_builder = builder.createSubgraphBuilder()
        else_builder.setGraphName("body_else")
        out_else = else_builder.aiOnnx.topk([inp, k_in], axis=0)[0]
        else_builder.addOutputTensor(out_else)

        o = builder.aiOnnx.logical_if([cond], 1, else_builder, then_builder)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        return [np.array([i1[np.argmax(i1)]] + (i1 * condition)).astype(np.int32)]

    op_tester.run(init_builder, reference, step_type="infer")
