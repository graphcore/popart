# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np

# pylint is disabled as op_tester is used as a fixture
from operators_test.conftest import op_tester  # pylint: disable=unused-import


def test_dont_inplace_output_consumers(op_tester):
    """

       Check that an Op, in a sugbraph, with an input which is a graph output,
       is not inplace. Example below : the Op with * cannot be inplaced.

       Input = [1,1]
         |
    ------------
    |  scale-2 |
    |    |     |
    |  scale-2 |
    |    |     |
    |  scale-2-|-------- add --- final output is [72, 72]
    |    |     |       /                        ([8, 8]  + [64, 64])
    | *scale-2 |      /
    |    |     |     /
    |  scale-2 |    /
    |    |     |   /
    |  scale-2-|--/
    |          |
    ------------

    """
    d0 = np.asarray([1.0, 1.0]).astype(np.float32)

    def get_init_builder():
        def init_builder(builder):
            i0 = builder.addInputTensor(d0)
            subgraph_builder = builder.createSubgraphBuilder()
            subgraph_builder.addInputTensorFromParentGraph(i0)
            i1 = subgraph_builder.aiGraphcore.scale([i0], 2.0, "hoop1")
            i2 = subgraph_builder.aiGraphcore.scale([i1], 2.0, "hoop2")
            i3 = subgraph_builder.aiGraphcore.scale([i2], 2.0, "hoop3")
            i4 = subgraph_builder.aiGraphcore.scale([i3], 2.0, "hoop4")
            i5 = subgraph_builder.aiGraphcore.scale([i4], 2.0, "hoop5")
            i6 = subgraph_builder.aiGraphcore.scale([i5], 2.0, "hoop5")
            subgraph_builder.addOutputTensor(i3)  # 8
            subgraph_builder.addOutputTensor(i6)  # 64
            outs = builder.aiGraphcore.call([i0], 2, subgraph_builder)
            summation = builder.aiOnnx.add([outs[0], outs[1]])
            builder.addOutputTensor(summation)
            return [summation]

        return init_builder

    def reference(_):  # ref_data is an unused argument
        return [np.array([72.0, 72.0]).astype(np.float32)]

    op_tester.setPatterns(["InPlace"], enableRuntimeAsserts=False)
    op_tester.run(get_init_builder(), reference, "infer")


def test_dont_inplace_when_aliased_inputs(op_tester):
    data = np.random.randn(5, 3, 16, 16).astype(np.float32)

    def init_builder(builder):
        x = builder.addInputTensor(data)
        # Inplacing of Concat results in a race condition in an already
        # inplace Add (AddLhsInPlace).
        concatenated = builder.aiOnnx.concat([x, x], 0)
        starts = builder.aiOnnx.constant(np.array([4]).astype(np.int32))
        ends = builder.aiOnnx.constant(np.array([9]).astype(np.int32))
        axes = builder.aiOnnx.constant(np.array([0]).astype(np.int32))
        sliced = builder.aiOnnx.slice([concatenated, starts, ends, axes])
        result = builder.aiOnnx.add([x, sliced])
        return [result]

    def reference(_):  # ref_data is an unused argument
        sliced = np.concatenate([data, data], axis=0)[4:9]
        return [data + sliced]

    op_tester.setPatterns(["InPlace"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")
