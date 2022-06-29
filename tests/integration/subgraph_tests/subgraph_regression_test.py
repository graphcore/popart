# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import numpy as np


def test_postnrepl_overzealous_elimination():

    # Reproducer for T36270, included as regression test. This test doesn't
    # actually do any assertions, it just checks that a code path that
    # previously failed does not result in any exceptions.
    #
    # The bug was that the PostNRepl pattern removed the gradient sum op that
    # produces Gradient_<in0> (which has 1 input) in the backwards subgraph, also
    # rewriting the subgraph itself to use the input to the gradient sum op
    # instead, as it's identical. However, the tensor produced by the op is a
    # graph output that is used by a call op in the main graph. The pattern did
    # not adjust this CallOp or the subgraph's output tensors and so the CallOp
    # in the main graph fails because it's using a tensor that no longer exists.

    def get_subgraph_builder(b, w):
        builder = b.createSubgraphBuilder()
        builder.addInputTensorFromParentGraph(w)

        in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT16", [4, 32, 16, 64]))

        x = builder.aiOnnx.matmul([in0, w])

        builder.addOutputTensor(x)
        return builder

    # building model and dataflow
    builder = popart.Builder()

    in0 = builder.addInputTensor(popart.TensorInfo('FLOAT16', [4, 32, 1, 64]),
                                 "in0")
    w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                          "weights")

    fn = get_subgraph_builder(builder, w)
    x = builder.aiGraphcore.call([w, in0], 1, fn)[0]
    l1_loss = builder.aiGraphcore.l1loss([x], 1.0)

    optimizer = popart.SGD({
        "defaultLearningRate": (0.1, False),
        "defaultWeightDecay": (0, True)
    })
    device = popart.DeviceManager().createIpuModelDevice({})

    # create training session
    popart.TrainingSession(fnModel=builder.getModelProto(),
                           loss=l1_loss,
                           deviceInfo=device,
                           optimizer=optimizer,
                           dataFlow=popart.DataFlow(1, {}),
                           userOptions=popart.SessionOptions())


def test_incomplete_grad():

    # Reproducer for T37001, included as regression test. This test doesn't
    # actually check any assertions, it just ensure that a code path that
    # previously failed does not result in any exceptions.
    #
    # The problem originally revealed by this test was that an exception was
    # thrown if for some inputs of a fwd subgraph the backwards pass creator was
    # not able to create gradients for those inputs (for example for a seed
    # input). This problem was fixed in the code base by allowing subgraph
    # inputs in the fwd subgraph to not have an associated gradients outputs in
    # the associated bwd subgraph.

    def get_subgraph_builder(builder, weights, labels):

        subgraph_builder = builder.createSubgraphBuilder()
        subgraph_builder.addInputTensorFromParentGraph(weights)
        input = subgraph_builder.addInputTensor(
            popart.TensorInfo("FLOAT16", [4, 32, 1, 64]))
        subgraph_builder.addInputTensorFromParentGraph(labels)

        matmul_out = subgraph_builder.aiOnnx.matmul([input, weights])
        log_probs = subgraph_builder.aiOnnx.logsoftmax([matmul_out], axis=3)
        log_probs_compact = subgraph_builder.aiOnnx.gather([log_probs, labels],
                                                           axis=3)
        subgraph_builder.addOutputTensor(log_probs_compact)

        return subgraph_builder

    builder = popart.Builder()

    float16_input = builder.addInputTensor(
        popart.TensorInfo("FLOAT16", [4, 32, 1, 64]), "float16_input")
    int32_input = builder.addInputTensor(popart.TensorInfo("INT32", [4, 2]),
                                         "int32_input")
    weights = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16),
                                                "weights")

    fn = get_subgraph_builder(builder, weights, int32_input)
    log_probs_compact = builder.aiGraphcore.call(
        [weights, float16_input, int32_input], 1, fn)[0]
    l1_loss = builder.aiGraphcore.l1loss([log_probs_compact], 1.0)

    optimizer = popart.SGD({
        "defaultLearningRate": (0.1, False),
        "defaultWeightDecay": (0, True)
    })

    _ = popart.TrainingSession(
        builder.getModelProto(),
        loss=l1_loss,
        deviceInfo=popart.DeviceManager().createIpuModelDevice({}),
        optimizer=optimizer,
        dataFlow=popart.DataFlow(1, {}),
        userOptions=popart.SessionOptions())
