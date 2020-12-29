# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import torch
from op_tester import op_tester


def test_simple_for_loop(op_tester):
    i1 = np.array([3]).astype(np.float32)
    i2 = np.array([6]).astype(np.float32)

    def get_init_builder(builder_setting):
        def init_builder(builder):
            # main graph
            builder.setGraphName("main_graph")
            a = builder.addInputTensor(i1)
            b = builder.addInputTensor(i2)
            M = builder.aiOnnx.constant(np.array(10).astype(np.int64), "M")
            cond = builder.aiOnnx.constant(
                np.array(True).astype(np.bool), "cond")

            # loop body subgraph
            loop_builder = builder.createSubgraphBuilder()
            loop_builder.setGraphName("body")

            # loop body inputs: [iteration_number, condition_in, lcd_tensors]
            loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
            keepgoing = loop_builder.addInputTensor(
                popart.TensorInfo("BOOL", []))
            a_in = loop_builder.addUntypedInputTensor(a)

            if builder_setting == "InPlace":
                a_out = loop_builder.aiOnnx.add([a_in, b])
            elif builder_setting == "NoInplace":
                a_out = loop_builder.aiOnnx.add([a_in, b])
            elif builder_setting == "Implicit":
                a_out = loop_builder.aiOnnx.add([b, a_in])
            else:
                a_out = loop_builder.aiOnnx.add([a_in, b])

            # loop body outputs: [condition_out, a_out]
            loop_builder.addOutputTensor(keepgoing)
            loop_builder.addOutputTensor(a_out)

            # Inputs: [iteration_number, condition_in, a_in]
            o = builder.aiOnnx.loop([M, cond, a], 1, loop_builder)[0]
            builder.addOutputTensor(o)
            return [o]

        return init_builder

    def reference(ref_data):
        class LoopModule(torch.nn.Module):
            def __init__(self):
                super(LoopModule, self).__init__()

            def forward(self, a, b, m: int):
                for i in range(m):
                    a = a + b
                return a

        model = torch.jit.script(LoopModule())
        a = torch.tensor([3], dtype=torch.float32)
        b = torch.tensor([6], dtype=torch.float32)
        m = torch.tensor(10, dtype=torch.long)
        output = model(a, b, m)
        return [output]

    op_tester.setPatterns(popart.PatternsLevel.NoPatterns,
                          enableRuntimeAsserts=False)
    # T23410: This test doesn't work with inplacing enabled.
    op_tester.inplacing = False
    op_tester.run(get_init_builder("NoInplace"), reference, step_type='infer')
    op_tester.run(get_init_builder("Inplace"), reference, step_type='infer')
    op_tester.run(get_init_builder("Implicit"), reference, step_type='infer')


def test_loop_matmul(op_tester):
    i1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    i2 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    trip_count = 10

    def init_builder(builder):
        builder.setGraphName("main_graph")
        a = builder.addInputTensor(i1)
        b = builder.addInputTensor(i2)

        M = builder.aiOnnx.constant(np.array(trip_count).astype(np.int64))
        cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond")

        loop_builder = builder.createSubgraphBuilder()
        loop_builder.setGraphName("body")
        # Inputs: [iteration_number, condition_in, a_in]
        loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
        keepgoing = loop_builder.addInputTensor(popart.TensorInfo("BOOL", []))
        a_in = loop_builder.addUntypedInputTensor(a)
        a_out = loop_builder.aiOnnx.matmul([a_in, b])

        # Outputs: [condition_out, a_out]
        loop_builder.addOutputTensor(keepgoing)
        loop_builder.addOutputTensor(a_out)

        o = builder.aiOnnx.loop([M, cond, a], 1, loop_builder)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = i1
        b = i2

        x = a
        for i in range(trip_count):
            x = np.matmul(x, b)

        return [x]

    op_tester.setPatterns(popart.PatternsLevel.NoPatterns,
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')
