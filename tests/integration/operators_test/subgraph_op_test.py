# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart


def test_nested_simple(op_tester):
    """
    Creates nested subgraph Ops (Call & Loop):
    
    Call
      Loop
        MatMul
        Add
        Add
    Call
      Loop
        MatMul
        Add
        Add
    Add
    
    and tests if using parent scope tensors inside subgraphs works correctly
    """

    i1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    i2 = np.array([[4, 3], [2, 1]]).astype(np.float32)
    i3 = np.array([[5, 6], [7, 8]]).astype(np.float32)
    i4 = np.array([[8, 7], [6, 5]]).astype(np.float32)
    trip_count = 6

    def init_builder(builder):
        builder.setGraphName("main_graph")
        a = builder.addInputTensor(i1)
        b = builder.addInitializedInputTensor(i2)
        c = builder.addInitializedInputTensor(i3)

        m = builder.aiOnnx.constant(
            np.array(trip_count).astype(np.int64), "m0")
        cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond0")

        call_builder = builder.createSubgraphBuilder()
        call_builder.setGraphName("call")

        loop_builder = call_builder.createSubgraphBuilder()
        loop_builder.setGraphName("loop")

        # d should be usable in the loop_builder, since it is a subgraph to
        # call_builder
        d = call_builder.aiOnnx.constant(i4, "d")

        # Inputs to loop
        iter = loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
        cond_in = loop_builder.addUntypedInputTensor(cond)
        a_in = loop_builder.addUntypedInputTensor(a)

        # Computation inside loop
        a_out = loop_builder.aiOnnx.matmul([a_in, b])
        a_out = loop_builder.aiOnnx.add([a_out, c])
        a_out = loop_builder.aiOnnx.add([a_out, d])

        # Outputs of loop
        loop_builder.addOutputTensor(cond_in)
        loop_builder.addOutputTensor(a_out)

        # Call
        call_loop = call_builder.aiOnnx.loop([m, cond, a], 1, loop_builder)[0]
        call_builder.addOutputTensor(call_loop)

        # Main graph
        o0 = builder.aiGraphcore.call([], 1, call_builder)[0]
        o1 = builder.aiGraphcore.call([], 1, call_builder)[0]
        o = builder.aiOnnx.add([o0, o1])

        builder.addOutputTensor(o)

        return [o]

    def reference(ref_data):
        a = i1
        b = i2
        c = i3
        d = i4

        x = a
        for i in range(trip_count):
            x = np.matmul(x, b)
            x = x + c
            x = x + d

        x = x * 2
        return [x]

    op_tester.run(init_builder, reference, step_type='infer')


def test_nested_complex(op_tester):
    """
    Creates nested subgraph Ops (Call & Loop & If):
    
    Loop
      Loop
          If
            MatMul
            Add
          Else
            Add
    Call
      Loop
          If
            MatMul
            Add
          Else
            Add
    Add
    
    and tests if using parent scope tensors inside subgraphs works correctly
    """

    i1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    i2 = np.array([[4, 3], [2, 1]]).astype(np.float32)
    i3 = np.array([[5, 6], [7, 8]]).astype(np.float32)
    i4 = np.array([[8, 7], [6, 5]]).astype(np.float32)
    i5 = np.array([[-1, -1], [-2, -2]]).astype(np.float32)
    i6 = np.array([[-1, 1], [-3, 3]]).astype(np.float32)
    trip_count_0 = 4
    trip_count_1 = 3
    if_else_count = 2

    def init_builder(builder):
        builder.setGraphName("main_graph")
        a = builder.addInputTensor(i1)
        b = builder.addInitializedInputTensor(i2)
        c = builder.addInitializedInputTensor(i3)
        d = builder.addInitializedInputTensor(i4)
        f = builder.addInputTensor(i6)

        m0 = builder.aiOnnx.constant(
            np.array(trip_count_0).astype(np.int64), "m0")
        cond0 = builder.aiOnnx.constant(
            np.array(True).astype(np.bool), "cond0")

        m1 = builder.aiOnnx.constant(
            np.array(trip_count_1).astype(np.int64), "m1")
        cond1 = builder.aiOnnx.constant(
            np.array(True).astype(np.bool), "cond1")

        m2 = builder.aiOnnx.constant(
            np.array(if_else_count).astype(np.int64), "m2")

        loop_builder0 = builder.createSubgraphBuilder()
        loop_builder0.setGraphName("l0")
        loop_builder1 = builder.createSubgraphBuilder()
        loop_builder1.setGraphName("l1")
        call_builder0 = builder.createSubgraphBuilder()
        call_builder0.setGraphName("c0")
        if_builder0 = loop_builder1.createSubgraphBuilder()
        if_builder0.setGraphName("i0")
        else_builder0 = loop_builder1.createSubgraphBuilder()
        else_builder0.setGraphName("e0")

        # Inputs to loop 1
        iter1 = loop_builder1.addInputTensor(popart.TensorInfo("INT64", []))
        cond_in1 = loop_builder1.addUntypedInputTensor(cond1)
        a_in1 = loop_builder1.addUntypedInputTensor(a)
        b_in1 = loop_builder1.addUntypedInputTensor(b)

        # Computation inside if
        a_out_if0 = if_builder0.aiOnnx.matmul([a_in1, b_in1])
        a_out_if0 = if_builder0.aiOnnx.add([a_out_if0, c])
        if_builder0.addOutputTensor(a_out_if0)

        # Computation inside else
        a_out_else0 = else_builder0.aiOnnx.add([a_in1, f])
        else_builder0.addOutputTensor(a_out_else0)

        # Computation inside loop 1
        cond_if = loop_builder1.aiOnnx.less([iter1, m2])
        a_out1 = loop_builder1.aiOnnx.logical_if(
            [cond_if],
            1,
            else_builder0,
            if_builder0,
        )[0]

        # Outputs of loop 1
        loop_builder1.addOutputTensor(cond_in1)
        loop_builder1.addOutputTensor(a_out1)
        loop_builder1.addOutputTensor(b_in1)

        # Call 0
        e = call_builder0.aiOnnx.constant(i5, "e")
        call0_loop1 = call_builder0.aiOnnx.loop([m1, cond1, a, e], 2,
                                                loop_builder1)[0]
        call_builder0.addOutputTensor(call0_loop1)

        # Loop 0
        iter0 = loop_builder0.addInputTensor(popart.TensorInfo("INT64", []))
        cond_in0 = loop_builder0.addUntypedInputTensor(cond0)
        a_in0 = loop_builder0.addUntypedInputTensor(a)
        loop0_loop1 = loop_builder0.aiOnnx.loop([m1, cond1, a_in0, d], 2,
                                                loop_builder1)[0]
        loop_builder0.addOutputTensor(cond_in0)
        loop_builder0.addOutputTensor(loop0_loop1)

        # Main graph
        loop0 = builder.aiOnnx.loop([m0, cond0, a], 1, loop_builder0)[0]
        call0 = builder.aiGraphcore.call([], 1, call_builder0)[0]

        o = builder.aiOnnx.add([loop0, call0])

        builder.addOutputTensor(o)

        return [o]

    def reference(ref_data):
        a = i1
        b = i2
        c = i3
        d = i4
        e = i5
        f = i6

        def fun(x, y):
            for j in range(trip_count_1):
                if j < if_else_count:
                    x = np.matmul(x, y)
                    x = x + c
                else:
                    x = x + f
            return x

        x = a
        for i in range(trip_count_0):
            x = fun(x, d)
        x += fun(a, e)
        return [x]

    op_tester.run(init_builder, reference, step_type='infer')
