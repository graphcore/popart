import numpy as np
import popart
import pytest
from op_tester import op_tester


#  Subgraph, sg:
#  in0  in1
#   |    |
#  Add --
#   |
#  out
#
#  Main graph:
#
#  in0  in1    in2
#   |    |      |
#  Call(sg)_0   |
#      |        |
#     act      /
#      |      /
#     Call(sg)_1
#         |
#        out
def test_call(op_tester):
    d0 = np.asarray([2, -1]).astype(np.int32)
    d1 = np.asarray([-4, 3]).astype(np.int32)
    d2 = np.asarray([1, 2]).astype(np.int32)

    def get_init_builder(input_tensor_method):
        def init_builder(builder):
            i0 = builder.addInputTensor(d0)
            i1 = builder.addInputTensor(d1)
            i2 = builder.addInputTensor(d2)

            subgraph_builder = builder.createSubgraphBuilder()

            if input_tensor_method == "untyped":
                sgi0 = subgraph_builder.addUntypedInputTensor()
                sgi1 = subgraph_builder.addUntypedInputTensor()
            elif input_tensor_method == "with_info":
                info = popart.TensorInfo("INT32", [2])
                sgi0 = subgraph_builder.addInputTensor(info)
                sgi1 = subgraph_builder.addInputTensor(info)
            elif input_tensor_method == "from_higher_scope":
                subgraph_builder.addInputTensorFromParentGraph(i0)
                subgraph_builder.addInputTensorFromParentGraph(i1)

            if input_tensor_method == "from_higher_scope":
                subgraph_builder.addOutputTensor(
                    subgraph_builder.aiOnnxOpset10.add([i0, i1]))
            else:
                subgraph_builder.addOutputTensor(
                    subgraph_builder.aiOnnxOpset10.add([sgi0, sgi1]))

            act = builder.aiGraphcore.call([i0, i1], 1, subgraph_builder)[0]
            out = builder.aiGraphcore.call([act, i2], 1, subgraph_builder)[0]
            builder.addOutputTensor(out)
            return [out]

        return init_builder

    def reference(ref_data):
        return [d0 + d1 + d2]

    op_tester.run(get_init_builder("untyped"), reference, 'infer')
    op_tester.run(get_init_builder("with_info"), reference, 'infer')
    op_tester.run(get_init_builder("from_higher_scope"), reference, 'infer')


#  Subgraph, sg1:      Subgraph, sg0
#  in0      in1         in
#   |        |          |
# Call(sg0)  |         Scale
#   |       /           |
#  act0    /           out
#   |     /
#  Add ---
#   |
#  out
#
#  Main graph:
#
#  in0  in1      in2
#   |    |        |
#  Call(sg1)_0  Call(sg0)_0
#      |          |
#     act0      act1
#      |        /
#     Call(sg)_1
#         |
#        out
def test_nested_calls(op_tester):
    d0 = np.asarray([2, -1]).astype(np.int32)
    d1 = np.asarray([-4, 3]).astype(np.int32)
    d2 = np.asarray([1, 2]).astype(np.int32)

    def init_builder(builder):
        # sg0
        sg0_builder = builder.createSubgraphBuilder()
        sg0_i0 = sg0_builder.addUntypedInputTensor()
        sg0_builder.addOutputTensor(
            sg0_builder.aiGraphcoreOpset1.scale([sg0_i0], 2))

        # sg1
        sg1_builder = builder.createSubgraphBuilder()
        sg1_i0 = sg1_builder.addUntypedInputTensor()
        sg1_i1 = sg1_builder.addUntypedInputTensor()
        sg1_act0 = sg1_builder.aiGraphcoreOpset1.call([sg1_i0], 1,
                                                      sg0_builder)[0]
        sg1_builder.addOutputTensor(
            sg1_builder.aiOnnxOpset10.add([sg1_act0, sg1_i1]))

        # main
        i0 = builder.addInputTensor(d0)
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        act0 = builder.aiGraphcore.call([i0, i1], 1, sg1_builder)[0]
        act1 = builder.aiGraphcore.call([i2], 1, sg0_builder)[0]
        out = builder.aiGraphcore.call([act0, act1], 1, sg1_builder)[0]
        builder.addOutputTensor(out)
        return [out]

    def reference(ref_data):
        def sg0(in0):
            return 2 * in0

        def sg1(in0, in1):
            return in1 + sg0(in0)

        return [sg1(sg1(d0, d1), sg0(d2))]

    op_tester.run(init_builder, reference, 'infer')


def test_subgraph_with_zero_outputs(op_tester):
    d0 = np.asarray([2, -1]).astype(np.int32)

    def init_builder(builder):
        # sg0 (has no output)
        sg0_builder = builder.createSubgraphBuilder()
        sg0_i0 = sg0_builder.addUntypedInputTensor()
        sg0_builder.aiGraphcoreOpset1.scale([sg0_i0], 2)

        # main
        i0 = builder.addInputTensor(d0)
        builder.aiGraphcore.call([i0], 1, sg0_builder)
        return "dummy"

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None, 'infer')
    assert e_info.value.args[
        0] == "CallOp subgraph requires at least one output."


def test_subgraph_call_mismatch0(op_tester):
    d0 = np.asarray([2, -1]).astype(np.int32)

    def init_builder(builder):
        # sg0 (has 1 output)
        sg0_builder = builder.createSubgraphBuilder()
        sg0_i0 = sg0_builder.addUntypedInputTensor()
        sg0_builder.addOutputTensor(
            sg0_builder.aiGraphcoreOpset1.scale([sg0_i0], 2))

        # main (expects 2 outputs)
        i0 = builder.addInputTensor(d0)
        return builder.aiGraphcore.call([i0], 2, sg0_builder, "debug")

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None, 'infer')
    assert e_info.value.args[
        0] == "For CallOp 'debug', number of outputs (2) does not match that of the callee subgraph (1)"


def test_subgraph_call_mismatch1(op_tester):
    d0 = np.asarray([2, -1]).astype(np.int32)

    def init_builder(builder):
        # sg0 (has 1 input)
        sg0_builder = builder.createSubgraphBuilder()
        sg0_i0 = sg0_builder.addUntypedInputTensor()
        sg0_builder.addOutputTensor(
            sg0_builder.aiGraphcoreOpset1.scale([sg0_i0], 2))

        # main (expects 3 inputs)
        i0 = builder.addInputTensor(d0)
        i1 = builder.addInputTensor(d0)
        i2 = builder.addInputTensor(d0)
        return builder.aiGraphcore.call([i0, i1, i2], 1, sg0_builder, "debug")

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None, 'infer')
    assert e_info.value.args[
        0] == "For CallOp 'debug', number of inputs (3) does not match that of the callee subgraph (1)"
