# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch


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
            cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond")

            # loop body subgraph
            loop_builder = builder.createSubgraphBuilder()
            loop_builder.setGraphName("body")

            # loop body inputs: [iteration_number, condition_in, lcd_tensors]
            loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
            keepgoing = loop_builder.addInputTensor(popart.TensorInfo("BOOL", []))
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

    def reference(_):  # ref_data is an unused argument
        class LoopModule(torch.nn.Module):
            def __init__(self):
                super(LoopModule, self).__init__()

            def forward(self, a, b, m: int):
                for _ in range(m):
                    a = a + b
                return a

        model = torch.jit.script(LoopModule())
        a = torch.tensor([3], dtype=torch.float32)
        b = torch.tensor([6], dtype=torch.float32)
        m = torch.tensor(10, dtype=torch.long)
        output = model(a, b, m)
        return [output]

    op_tester.setPatterns(popart.PatternsLevel.NoPatterns, enableRuntimeAsserts=False)
    op_tester.run(get_init_builder("NoInplace"), reference, step_type="infer")
    op_tester.run(get_init_builder("Inplace"), reference, step_type="infer")
    op_tester.run(get_init_builder("Implicit"), reference, step_type="infer")


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
        tmp = loop_builder.aiOnnx.matmul([a_in, b])
        a_out = loop_builder.aiOnnx.matmul([tmp, b])

        # Outputs: [condition_out, a_out]
        loop_builder.addOutputTensor(keepgoing)
        loop_builder.addOutputTensor(a_out)

        o = builder.aiOnnx.loop([M, cond, a], 1, loop_builder)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = i1
        b = i2

        x = a
        for _ in range(trip_count):
            x = np.matmul(np.matmul(x, b), b)

        return [x]

    op_tester.setPatterns(popart.PatternsLevel.NoPatterns, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="infer")


def test_loop_stop(op_tester):
    i1 = np.array([1]).astype(np.float32)
    i2 = np.array([4]).astype(np.float32)
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
        _ = loop_builder.addUntypedInputTensor(cond)
        a_in = loop_builder.addUntypedInputTensor(a)
        a_out = loop_builder.aiOnnx.mul([a_in, b])
        max = loop_builder.aiOnnx.constant(np.array([128]).astype(np.float32), "max")
        keepgoing = loop_builder.aiOnnx.less([a_out, max])

        # Outputs: [condition_out, a_out]
        loop_builder.addOutputTensor(keepgoing)
        loop_builder.addOutputTensor(a_out)

        o = builder.aiOnnx.loop([M, cond, a], 1, loop_builder)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = i1
        b = i2

        x = a
        for _ in range(trip_count):
            x = x * b
            if x >= 128:
                break

        return [x]

    op_tester.setPatterns(popart.PatternsLevel.NoPatterns, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="infer")


def test_loop_scanout(op_tester):
    i1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    i2 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    i3 = np.array([[-1, -2], [-3, -4]]).astype(np.float32)
    trip_count = 10

    def init_builder(builder):
        builder.setGraphName("main_graph")
        a = builder.addInputTensor(i1)
        b = builder.addInputTensor(i2)
        c = builder.addInputTensor(i3)

        M = builder.aiOnnx.constant(np.array(trip_count).astype(np.int64))
        cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond")

        loop_builder = builder.createSubgraphBuilder()
        loop_builder.setGraphName("body")
        # Inputs: [iteration_number, condition_in, a_in]
        loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
        keepgoing = loop_builder.addInputTensor(popart.TensorInfo("BOOL", []))
        a_in = loop_builder.addUntypedInputTensor(a)
        a_out = loop_builder.aiOnnx.matmul([a_in, b])
        d_out = loop_builder.aiOnnx.add([a_out, c])

        # Outputs: [condition_out, a_out]
        loop_builder.addOutputTensor(keepgoing)
        # Loop carried dependency
        loop_builder.addOutputTensor(a_out)
        # Loop scan outputs
        loop_builder.addOutputTensor(a_out)
        loop_builder.addOutputTensor(d_out)

        o, co0, co1 = builder.aiOnnx.loop([M, cond, a], 3, loop_builder)
        builder.addOutputTensor(o)
        builder.addOutputTensor(co0)
        builder.addOutputTensor(co1)
        return [o, co0, co1]

    def reference(_):  # ref_data is an unused argument
        a = i1
        b = i2
        c = i3

        x = a
        scanx = []
        scany = []
        for _ in range(trip_count):
            x = np.matmul(x, b)
            y = np.add(x, c)
            scanx.append(x)
            scany.append(y)

        return [x, np.asarray(scanx), np.asarray(scany)]

    op_tester.run(init_builder, reference, step_type="infer")


def test_loop_const_implicit_tensor(op_tester):
    i1 = np.array([4]).astype(np.float32)
    trip_count = 10

    def init_builder(builder):
        builder.setGraphName("main_graph")
        a = builder.addInputTensor(i1)
        M = builder.aiOnnx.constant(np.array(trip_count).astype(np.int64))
        cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond")

        k_in = builder.aiOnnx.constant(np.array([1]).astype(np.int64), "k")

        loop_builder = builder.createSubgraphBuilder()
        loop_builder.setGraphName("body")
        # Inputs: [iteration_number, condition_in, a_in]
        loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
        keepgoing = loop_builder.addInputTensor(popart.TensorInfo("BOOL", []))
        a_in = loop_builder.addUntypedInputTensor(a)
        topk = loop_builder.aiOnnx.topk([a_in, k_in], axis=0)[0]

        # Outputs: [condition_out, a_out]
        loop_builder.addOutputTensor(keepgoing)
        loop_builder.addOutputTensor(topk)

        o = builder.aiOnnx.loop([M, cond, a], 1, loop_builder)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        return [np.array([i1[np.argmax(i1)]]).astype(np.float32)]

    op_tester.run(init_builder, reference, step_type="infer")
