# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import torch
import torch.nn as nn

import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

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


@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_call(op_tester, subgraphCopyingStrategy):
    op_tester.options.subgraphCopyingStrategy = subgraphCopyingStrategy
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
                    subgraph_builder.aiOnnx.add([i0, i1]))
            else:
                subgraph_builder.addOutputTensor(
                    subgraph_builder.aiOnnx.add([sgi0, sgi1]))

            act = builder.aiGraphcore.call([i0, i1], 1, subgraph_builder)[0]
            out = builder.aiGraphcore.call([act, i2], 1, subgraph_builder)[0]
            builder.addOutputTensor(out)
            return [out]

        return init_builder

    def reference(_):  # ref_data is an unused argument
        return [d0 + d1 + d2]

    op_tester.run(get_init_builder("untyped"), reference, 'infer')
    op_tester.run(get_init_builder("with_info"), reference, 'infer')
    op_tester.run(get_init_builder("from_higher_scope"), reference, 'infer')


#  Subgraph, sg:
#  in0    in1
#   |      |
#  MatMul --
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
@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_call_grad_1(op_tester, subgraphCopyingStrategy):
    op_tester.options.subgraphCopyingStrategy = subgraphCopyingStrategy
    shape = [4, 4]
    d0 = np.random.normal(size=shape).astype(np.float32)
    d1 = np.random.normal(size=shape).astype(np.float32)
    d2 = np.random.normal(size=shape).astype(np.float32)

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
                info = popart.TensorInfo("FLOAT", shape)
                sgi0 = subgraph_builder.addInputTensor(info)
                sgi1 = subgraph_builder.addInputTensor(info)
            elif input_tensor_method == "from_higher_scope":
                subgraph_builder.addInputTensorFromParentGraph(i0)
                subgraph_builder.addInputTensorFromParentGraph(i1)

            if input_tensor_method == "from_higher_scope":
                subgraph_builder.addOutputTensor(
                    subgraph_builder.aiOnnx.matmul([i0, i1]))
            else:
                subgraph_builder.addOutputTensor(
                    subgraph_builder.aiOnnx.matmul([sgi0, sgi1]))

            act = builder.aiGraphcore.call([i0, i1], 1, subgraph_builder)[0]
            out = builder.aiGraphcore.call([act, i2], 1, subgraph_builder)[0]

            builder.addOutputTensor(out)
            return [
                out,
                popart.reservedGradientPrefix() + out,
                popart.reservedGradientPrefix() + i0,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + i2
            ]

        return init_builder

    def reference(ref_data):
        t0 = torch.tensor(d0, requires_grad=True)
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        o = torch.chain_matmul(t0, t1, t2)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))

        return [o, d__o, t0.grad, t1.grad, t2.grad]

    op_tester.setPatterns(popart.PatternsLevel.Default,
                          enableRuntimeAsserts=False)
    op_tester.run(get_init_builder("untyped"), reference, 'train')
    op_tester.run(get_init_builder("with_info"), reference, 'train')
    op_tester.run(get_init_builder("from_higher_scope"), reference, 'train')


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
@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_nested_calls(op_tester, subgraphCopyingStrategy):
    op_tester.options.subgraphCopyingStrategy = subgraphCopyingStrategy
    d0 = np.asarray([2, -1]).astype(np.int32)
    d1 = np.asarray([-4, 3]).astype(np.int32)
    d2 = np.asarray([1, 2]).astype(np.int32)

    def init_builder(builder):
        # sg0
        sg0_builder = builder.createSubgraphBuilder()
        sg0_i0 = sg0_builder.addUntypedInputTensor()
        sg0_builder.addOutputTensor(sg0_builder.aiGraphcore.scale([sg0_i0], 2))

        # sg1
        sg1_builder = builder.createSubgraphBuilder()
        sg1_i0 = sg1_builder.addUntypedInputTensor()
        sg1_i1 = sg1_builder.addUntypedInputTensor()
        sg1_act0 = sg1_builder.aiGraphcore.call([sg1_i0], 1, sg0_builder)[0]
        sg1_builder.addOutputTensor(sg1_builder.aiOnnx.add([sg1_act0, sg1_i1]))

        # main
        i0 = builder.addInputTensor(d0)
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        act0 = builder.aiGraphcore.call([i0, i1], 1, sg1_builder)[0]
        act1 = builder.aiGraphcore.call([i2], 1, sg0_builder)[0]
        out = builder.aiGraphcore.call([act0, act1], 1, sg1_builder)[0]
        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        def sg0(in0):
            return 2 * in0

        def sg1(in0, in1):
            return in1 + sg0(in0)

        return [sg1(sg1(d0, d1), sg0(d2))]

    op_tester.run(init_builder, reference, 'infer')


@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_subgraph_with_zero_outputs(op_tester, subgraphCopyingStrategy):
    op_tester.options.subgraphCopyingStrategy = subgraphCopyingStrategy
    d0 = np.asarray([2, -1]).astype(np.int32)

    def init_builder(builder):
        # sg0 (has no output)
        sg0_builder = builder.createSubgraphBuilder()
        sg0_i0 = sg0_builder.addUntypedInputTensor()
        sg0_builder.aiGraphcore.scale([sg0_i0], 2)

        # main
        i0 = builder.addInputTensor(d0)
        builder.aiGraphcore.call([i0], 1, sg0_builder)
        return "dummy"

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None, 'infer')
    assert e_info.value.args[
        0] == "For CallOp '', number of outputs (1) does not match that of the callee subgraph (0)"


@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_subgraph_call_mismatch0(op_tester, subgraphCopyingStrategy):
    op_tester.options.subgraphCopyingStrategy = subgraphCopyingStrategy
    d0 = np.asarray([2, -1]).astype(np.int32)

    def init_builder(builder):
        # sg0 (has 1 output)
        sg0_builder = builder.createSubgraphBuilder()
        sg0_i0 = sg0_builder.addUntypedInputTensor()
        sg0_builder.addOutputTensor(sg0_builder.aiGraphcore.scale([sg0_i0], 2))

        # main (expects 2 outputs)
        i0 = builder.addInputTensor(d0)
        return builder.aiGraphcore.call([i0], 2, sg0_builder, "debug")

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None, 'infer')
    assert e_info.value.args[
        0] == "For CallOp 'debug', number of outputs (2) does not match that of the callee subgraph (1)"


@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_subgraph_call_mismatch1(op_tester, subgraphCopyingStrategy):
    op_tester.options.subgraphCopyingStrategy = subgraphCopyingStrategy
    d0 = np.asarray([2, -1]).astype(np.int32)

    def init_builder(builder):
        # sg0 (has 1 input)
        sg0_builder = builder.createSubgraphBuilder()
        sg0_i0 = sg0_builder.addUntypedInputTensor()
        sg0_builder.addOutputTensor(sg0_builder.aiGraphcore.scale([sg0_i0], 2))

        # main (expects 3 inputs)
        i0 = builder.addInputTensor(d0)
        i1 = builder.addInputTensor(d0)
        i2 = builder.addInputTensor(d0)
        return builder.aiGraphcore.call([i0, i1, i2], 1, sg0_builder, "debug")

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None, 'infer')
    assert e_info.value.args[
        0] == "For CallOp 'debug', number of inputs (3) does not match that of the callee subgraph (1)"


#  Subgraph, sg:
#  in0    in1
#   |      |
#  Matmul --
#   |
#  out
#
#  Main graph:
#
#  in0  in1
#   |    |
#  Call(sg)_0
#      |
#     act
@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_call_grad_2(op_tester, subgraphCopyingStrategy):
    op_tester.options.subgraphCopyingStrategy = subgraphCopyingStrategy
    d0 = np.random.normal(size=[4, 4]).astype(np.float32)
    d1 = np.random.normal(size=[4, 4]).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(d0, "input_d0")
        i1 = builder.addInputTensor(d1, "input_d1")

        subgraph_builder = builder.createSubgraphBuilder()

        subgraph_builder.addInputTensorFromParentGraph(i0)
        subgraph_builder.addInputTensorFromParentGraph(i1)

        subgraph_builder.addOutputTensor(
            subgraph_builder.aiOnnx.matmul([i0, i1], "add_inside_subgraph"))
        out = builder.aiGraphcore.call([i0, i1], 1, subgraph_builder,
                                       "call_subgraph")[0]

        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + out,
            popart.reservedGradientPrefix() + i0,
            popart.reservedGradientPrefix() + i1,
        ]

    def reference(ref_data):
        d0_t = torch.tensor(d0, requires_grad=True)
        d1_t = torch.tensor(d1, requires_grad=True)

        r = torch.matmul(d0_t, d1_t)

        r__o = ref_data.getOutputTensorGrad(0)
        r.backward(torch.Tensor(r__o))

        return [r, r__o, d0_t.grad, d1_t.grad]

    op_tester.setPatterns(popart.PatternsLevel.Default,
                          enableRuntimeAsserts=False)

    op_tester.run(init_builder, reference, 'train')


@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_call_grad_3(subgraphCopyingStrategy):
    # Generate some random input data
    trainingData = np.random.rand(1, 2).astype(np.float16)
    trainingDataLables = np.random.rand(1).astype(np.int32)

    # Create a builder and construct a graph
    def run_simple_model(subgraph=False):
        builder = popart.Builder()

        data_shape = popart.TensorInfo("FLOAT16", [1, 2])
        lbl_shape = popart.TensorInfo("INT32", [1])

        ip = builder.addInputTensor(data_shape)
        lb = builder.addInputTensor(lbl_shape)

        w = builder.addInitializedInputTensor(np.ones([2, 2], np.float16))
        b = builder.addInitializedInputTensor(np.ones([2], np.float16))
        gemm = builder.aiOnnx.gemm([ip, w, b], 1., 1., False, False)
        relu = builder.aiOnnx.relu([gemm])

        if subgraph:
            subgraph_builder = builder.createSubgraphBuilder()

            subgraph_builder.addInputTensorFromParentGraph(relu)
            sm = subgraph_builder.aiOnnx.softmax([relu])
            subgraph_builder.addOutputTensor(sm)

            call = builder.aiGraphcore.call([relu], 1, subgraph_builder,
                                            "call_subgraph")[0]
            nll = builder.aiGraphcore.nllloss([call, lb])
        else:
            sm = builder.aiOnnx.softmax([relu])
            nll = builder.aiGraphcore.nllloss([sm, lb])

        art = popart.AnchorReturnType("All")
        dataFlow = popart.DataFlow(
            1, {
                ip: art,
                popart.reservedGradientPrefix() + ip: art,
                relu: art,
                popart.reservedGradientPrefix() + relu: art,
                gemm: art,
                popart.reservedGradientPrefix() + gemm: art
            })

        trainingOptions = popart.SessionOptions()
        trainingOptions.subgraphCopyingStrategy = subgraphCopyingStrategy
        trainingSession = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=dataFlow,
            loss=nll,
            optimizer=popart.ConstSGD(0.001),
            userOptions=trainingOptions,
            deviceInfo=tu.create_test_device(),
            patterns=popart.Patterns(popart.PatternsLevel.Default))

        # Compile graph
        trainingSession.prepareDevice()

        # Execute the training graph
        # Create buffers to receive results from the execution
        trainingAnchors = trainingSession.initAnchorArrays()
        trainingStepio = popart.PyStepIO(
            {
                ip: trainingData,
                lb: trainingDataLables
            }, trainingAnchors)

        # Copy the weights to the device from the host
        trainingSession.weightsFromHost()

        # Run the training graph
        trainingSession.run(trainingStepio)

        # Copy the weights to the host from the device
        trainingSession.weightsToHost()
        return trainingAnchors

    sg_true = run_simple_model(True)
    sg_false = run_simple_model(False)
    for k1, k2 in zip(sg_true.keys(), sg_false.keys()):
        print(k1, k2)
        assert np.allclose(sg_true[k1], sg_false[k2])


def test_call_grad_scoped(op_tester):
    d0 = np.random.normal(size=[4, 4]).astype(np.float32)
    d1 = np.random.normal(size=[4, 4]).astype(np.float32)
    c0 = np.ones(shape=[4, 4]).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(d0, "input_d0")
        i1 = builder.addInputTensor(d1, "input_d1")

        subgraph_builder = builder.createSubgraphBuilder()

        subgraph_builder.addInputTensorFromParentGraph(i0)
        subgraph_builder.addInputTensorFromParentGraph(i1)
        # Put in a sub-scope just to mix things up.
        with subgraph_builder.nameScope("example_scope"):
            m = subgraph_builder.aiOnnx.matmul([i0, i1], "mul_inside_subgraph")
            # Add a constant - testing T15991
            c = subgraph_builder.aiOnnx.constant(c0, "subgraph_const")
            a = subgraph_builder.aiOnnx.add([m, c], "add_inside_subgraph")
            subgraph_builder.addOutputTensor(a)
        out = builder.aiGraphcore.call([i0, i1], 1, subgraph_builder,
                                       "call_subgraph")[0]

        builder.addOutputTensor(out)
        return [
            out,
            popart.reservedGradientPrefix() + out,
            popart.reservedGradientPrefix() + i0,
            popart.reservedGradientPrefix() + i1,
        ]

    def reference(ref_data):
        d0_t = torch.tensor(d0, requires_grad=True)
        d1_t = torch.tensor(d1, requires_grad=True)
        d1_t = torch.tensor(d1, requires_grad=True)
        r = torch.matmul(d0_t, d1_t)
        c_t = torch.tensor(c0, requires_grad=False)
        r = torch.add(r, c_t)

        # print(r)
        r__o = ref_data.getOutputTensorGrad(0)
        r.backward(torch.Tensor(r__o))

        return [r, r__o, d0_t.grad, d1_t.grad]

    op_tester.setPatterns(popart.PatternsLevel.Default,
                          enableRuntimeAsserts=False)

    op_tester.run(init_builder, reference, 'train')


# Model: N matmuls in series, sharing weights with a subgraph
#
#  Subgraph, sg:
#            w0
#  in        |
#   |        |
#  Matmul ---
#   |
#  ReLu
#   |
#  out
#
# Using the same w0 for each subgraph
#
#  w0 ---------------------------------
#       \           \                  \
#        Call0[sg] - Call1[sg] -  ... - CallN-1[sg] - out
#        /
#  in0 --
#
# Test:


@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_stacked_subgraphs(op_tester, subgraphCopyingStrategy):
    op_tester.options.subgraphCopyingStrategy = subgraphCopyingStrategy
    np.random.seed(0)
    numLayers = 4
    shape = [4, 4]
    input_ = np.random.rand(*shape).astype('float32')
    w0_init = np.random.rand(*shape).astype('float32')

    def init_builder(builder):
        in0 = builder.addInputTensor(input_)
        w0 = builder.addInitializedInputTensor(w0_init)

        subgraph_builder = builder.createSubgraphBuilder()
        info = popart.TensorInfo("FLOAT", shape)
        sgi0 = subgraph_builder.addInputTensor(info)
        sgi1 = subgraph_builder.addInputTensor(info)
        matmul = subgraph_builder.aiOnnx.matmul([sgi0, sgi1], "mm_layer")
        relu = subgraph_builder.aiOnnx.relu([matmul], "relu_layer")
        subgraph_builder.addOutputTensor(relu)

        actIn = in0
        for layer in range(numLayers):
            actIn = builder.aiGraphcore.call([actIn, w0], 1, subgraph_builder,
                                             f"subgraph_{layer}")[0]

        builder.addOutputTensor(actIn)
        return [
            actIn,
            popart.reservedGradientPrefix() + actIn,
            popart.reservedGradientPrefix() + in0,
            popart.reservedGradientPrefix() + w0,
        ]

    def reference(ref_data):
        in_t = torch.tensor(input_, requires_grad=True)
        w_t = nn.Parameter(torch.tensor(w0_init))

        r = in_t
        for _ in range(numLayers):
            r = torch.matmul(r, w_t)
            r = torch.relu(r)

        r__o = ref_data.getOutputTensorGrad(0)
        r.backward(torch.Tensor(r__o))

        return [r, r__o, in_t.grad, w_t.grad]

    op_tester.setPatterns(popart.PatternsLevel.Default,
                          enableRuntimeAsserts=False)

    op_tester.run(init_builder, reference, 'train')


@pytest.mark.parametrize("subgraphCopyingStrategy", [
    popart.SubgraphCopyingStrategy.OnEnterAndExit,
    popart.SubgraphCopyingStrategy.JustInTime
])
def test_stacked_subgraphs_2(subgraphCopyingStrategy):
    np.random.seed(0)
    shape = [4, 4]
    input_ = np.random.rand(*shape).astype('float32')
    w0_init = np.random.rand(*shape).astype('float32')

    def get_model(subgraph=False, steps=1):
        builder = popart.Builder()

        numLayers = 4
        in0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape))

        w0 = builder.addInitializedInputTensor(w0_init)

        if subgraph:
            subgraph_builder = builder.createSubgraphBuilder()
            info = popart.TensorInfo("FLOAT", shape)
            sgi0 = subgraph_builder.addInputTensor(info)
            sgi1 = subgraph_builder.addInputTensor(info)
            matmul = subgraph_builder.aiOnnx.matmul([sgi0, sgi1], "mm_layer")
            relu = subgraph_builder.aiOnnx.relu([matmul], "relu_layer")
            subgraph_builder.addOutputTensor(relu)

            actIn = in0
            for layer in range(numLayers):
                actIn = builder.aiGraphcore.call(
                    [actIn, w0], 1, subgraph_builder, f"subgraph_{layer}")[0]
        else:
            actIn = in0
            for layer in range(numLayers):
                actIn = builder.aiOnnx.matmul([actIn, w0],
                                              "mm_layer" + str(layer))
                actIn = builder.aiOnnx.relu([actIn], "relu_layer" + str(layer))

        actIn = builder.aiGraphcore.identityloss([actIn])
        builder.addOutputTensor(actIn)
        art = popart.AnchorReturnType("All")
        anchor_returns = {
            w0: art,
            popart.reservedGradientPrefix() + w0: art,
            in0: art,
            popart.reservedGradientPrefix() + in0: art
        }
        opts = popart.SessionOptions()
        opts.subgraphCopyingStrategy = subgraphCopyingStrategy
        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFlow=popart.DataFlow(
                                             1, anchor_returns),
                                         deviceInfo=tu.create_test_device(),
                                         optimizer=popart.ConstSGD(0.1),
                                         loss=actIn,
                                         userOptions=opts)

        anchors = session.initAnchorArrays()

        inputs = {in0: input_}
        stepio = popart.PyStepIO(inputs, anchors)

        session.prepareDevice()
        session.weightsFromHost()

        for _ in range(steps):
            session.run(stepio)
        return anchors

    sg_true = get_model(True, 5)
    sg_false = get_model(False, 5)
    for k1, k2 in zip(sg_true.keys(), sg_false.keys()):
        assert np.allclose(sg_true[k1], sg_false[k2])


# I used the example below for implementing subgraph partitioning, so it is
# useful to keep, but it doesn't actually test subgraph partitioning properly,
# so have disabled.
@pytest.mark.skip(reason="Test not required")
def test_subgraph_partitioning(op_tester):
    """
      What we're trying to achieve:
      main:
        a = 1                     # a = 1
        b = Call{'Callee':0}(a)   # b = 4 * (2 + a) = 12
        c = Call{'Callee':0}(b)   # c = 4 * (2 + b) = 62
        d = Call{'Callee':1}(a,c) # d = a + c = 63
      subgraph0(x0) -> 4 * subgraph1(2, x0):
        a0 = 2
        a1 = Call{'Callee':1}(a0, x0)
        out0 = 4 * a1
      subgraph1(x1,y1) -> x1 + y1:
        out1 = x1 + y1
    """

    da = np.asarray([1]).astype(np.int32)

    def init_builder(builder):
        # subgraph1
        sg1_builder = builder.createSubgraphBuilder()
        x1 = sg1_builder.addUntypedInputTensor()
        y1 = sg1_builder.addUntypedInputTensor()
        out1 = sg1_builder.aiOnnx.add([x1, y1])
        sg1_builder.addOutputTensor(out1)

        # subgraph0
        sg0_builder = builder.createSubgraphBuilder()
        x0 = sg0_builder.addUntypedInputTensor()
        a0 = sg0_builder.aiOnnx.constant(np.asarray([2]).astype(np.int32))
        a1 = sg0_builder.aiGraphcore.call([a0, x0], 1, sg1_builder)[0]
        out0 = sg0_builder.aiGraphcore.scale([a1], 4)
        sg0_builder.addOutputTensor(out0)

        # main
        a = builder.addInputTensor(da)
        b = builder.aiGraphcore.call([a], 1, sg0_builder)[0]
        c = builder.aiGraphcore.call([b], 1, sg0_builder)[0]
        d = builder.aiGraphcore.call([a, c], 1, sg1_builder)[0]
        builder.addOutputTensor(d)
        return [d]

    def reference(_):  # ref_data is an unused argument
        def sg1(x1, y1):
            return x1 + y1

        def sg0(x0):
            return sg1(2, x0) * 4

        return [sg1(da, sg0(sg0(da)))]

    op_tester.run(init_builder, reference, 'infer')


# TODO: uncomment-out test when T15830 is complete
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
# def test_empty_grad_subgraph():
#     builder = popart.Builder()
#     i0 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 4]))
#     i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 4]))
#
#     subgraph_builder = builder.createSubgraphBuilder()
#     subgraph_builder.addInputTensorFromParentGraph(i0)
#     subgraph_builder.addInputTensorFromParentGraph(i1)
#     mm = subgraph_builder.aiOnnx.matmul([i0, i1])
#     subgraph_builder.addOutputTensor(subgraph_builder.aiOnnx.dropout([mm], 1)[0])
#
#     out = builder.aiGraphcore.call([i0, i1], 1, subgraph_builder)[0]
#
#     anchorMap = {
#         popart.reservedGradientPrefix() + i0: popart.AnchorReturnType("All"),
#         popart.reservedGradientPrefix() + i1: popart.AnchorReturnType("All")
#     }
#
#     # This should throw some exception, as the grad subgraph is empty
#     session = popart.TrainingSession(
#       fnModel=builder.getModelProto(),
#       dataFlow=popart.DataFlow(1, anchorMap),
#       deviceInfo=tu.create_test_device(),
#       optimizer=popart.ConstSGD(0.1),
#       losses=[popart.IdentityLoss(out, out+"/loss")])
