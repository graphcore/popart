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
