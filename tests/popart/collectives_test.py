# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu
import pprint


@tu.requires_ipu
def test_replicated_allreduce():
    input_data = np.array(range(10), dtype=np.float32)
    replicatedGraphCount = 2
    builder = popart.Builder()
    t = builder.addInitializedInputTensor(input_data, "input")
    o = builder.aiGraphcore.replicatedallreduce([t])
    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = replicatedGraphCount
    numIpus = 2

    device = tu.create_test_device(numIpus=numIpus)
    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      userOptions=opts,
                                      deviceInfo=device)

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    ground_truth = 2.0 * np.array(range(10), dtype=np.float32)
    for i in range(replicatedGraphCount):
        assert np.allclose(anchors[o][i], ground_truth)


@tu.requires_ipu
def test_gcl_comm_group_attrs():
    input_data = np.array(range(10), dtype=np.float32)
    builder = popart.Builder()
    t = builder.addInitializedInputTensor(input_data, "input")
    o1 = builder.aiGraphcore.replicatedallreduce(
        [t], commGroup=[popart.CommGroupType.All, 4])
    with builder.commGroup(popart.CommGroupType.Consecutive, 4):
        o2 = builder.aiGraphcore.replicatedallreduce([t])
    with builder.commGroup(popart.CommGroupType.Orthogonal, 4):
        o3 = builder.aiGraphcore.replicatedallreduce([t])
    builder.addOutputTensor(o1)
    builder.addOutputTensor(o2)
    builder.addOutputTensor(o3)
    attributes = builder.getAllNodeAttributeNames(set([o2]))
    assert "__collectiveCommGroup" in attributes
    attr = builder.getInt64VectorNodeAttribute("__collectiveCommGroup",
                                               set([o2]))
    assert attr == [int(popart.CommGroupType.Consecutive), 4]

    attributes = builder.getAllNodeAttributeNames(set([o3]))
    assert "__collectiveCommGroup" in attributes

    attr = builder.getInt64VectorNodeAttribute("__collectiveCommGroup",
                                               set([o3]))

    assert attr == [int(popart.CommGroupType.Orthogonal), 4]
