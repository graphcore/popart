# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import test_util as tu
import pytest


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

    with tu.create_test_device(numIpus=numIpus) as device:
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
@pytest.mark.parametrize(
    "commGroupAndVarSettings",
    [(None, False),
     (popart.CommGroup(popart.CommGroupType.Consecutive, 2), False),
     (popart.CommGroup(popart.CommGroupType.Consecutive, 2), True)])
def test_replicated_reducescatter(commGroupAndVarSettings):
    commGroup = commGroupAndVarSettings[0]
    useVarSettings = commGroupAndVarSettings[1]

    replicatedGraphCount = 4
    numIpus = replicatedGraphCount
    replicas_per_group = replicatedGraphCount
    num_groups = 1
    num_shards = replicatedGraphCount
    if commGroup != None:
        num_shards = replicatedGraphCount // commGroup.replicaGroupSize
        if useVarSettings:
            num_groups = commGroup.replicaGroupSize
            replicas_per_group = replicatedGraphCount // num_groups

    input_data = np.array(range(12 * num_groups), dtype=np.float32)

    if useVarSettings:
        input_data = input_data.reshape([num_groups, -1])

    builder = popart.Builder()

    if useVarSettings:
        variableSetting = popart.VariableSettings(commGroup)
        builder.embedReplicationFactor(replicatedGraphCount)
    else:
        variableSetting = popart.VariableSettings()

    t = builder.addInitializedInputTensor(input_data, variableSetting, "input")
    o = builder.aiGraphcore.replicatedreducescatter([t], commGroup=commGroup)
    builder.addOutputTensor(t)
    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = replicatedGraphCount

    with tu.create_test_device(numIpus=numIpus) as device:
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        inputs = {}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

    ground_truth = num_shards * input_data.reshape(
        [num_groups, num_shards, -1])

    print(ground_truth)
    print(anchors[o])

    for i in range(num_groups):
        for j in range(replicas_per_group):
            assert np.allclose(
                anchors[o][i * replicas_per_group + j].flatten(),
                ground_truth[i][j % num_shards].flatten())


@tu.requires_ipu
def test_gcl_comm_group_attrs():
    input_data = np.array(range(10), dtype=np.float32)
    builder = popart.Builder()
    t = builder.addInitializedInputTensor(input_data, "input")
    o1 = builder.aiGraphcore.replicatedallreduce([t],
                                                 commGroup=popart.CommGroup(
                                                     popart.CommGroupType.All,
                                                     4))
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
