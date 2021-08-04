# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import json


def test_remove_redundant_identityloss():
    """
    Verify that an identityloss applied to a scalar tenosr (and its
    corresponding grad op) is pruned
    """
    builder = popart.Builder()
    in0 = builder.addInputTensor("FLOAT", [1, 4, 4])
    in1 = builder.addInputTensor("INT32", [1, 4])
    w0 = builder.addInitializedInputTensor(
        np.random.rand(4, 4).astype(np.float32))
    m = builder.aiOnnx.matmul([in0, w0])
    l1 = builder.aiGraphcore.nllloss([m, in1],
                                     reduction=popart.ReductionType.Sum)
    idloss = builder.aiGraphcore.identityloss([l1])

    def getGraph(applyOpToIdentityPattern):
        patterns = popart.Patterns(popart.PatternsLevel.All)
        patterns.OpToIdentity = applyOpToIdentityPattern

        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            deviceInfo=popart.DeviceManager().createCpuDevice(),
            dataFlow=popart.DataFlow(1, [idloss]),
            loss=idloss,
            optimizer=popart.ConstSGD(0.1),
            patterns=patterns)
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        return ir['maingraph']

    graph = getGraph(False)
    idlossops = [op for op in graph if op['type'] == 'IdentityLoss']
    idlossgradops = [op for op in graph if op['type'] == 'IdentityLossGrad']
    assert len(idlossops) == 1
    assert len(idlossgradops) == 1

    graph = getGraph(True)
    idlossops = [op for op in graph if op['type'] == 'IdentityLoss']
    idlossgradops = [op for op in graph if op['type'] == 'IdentityLossGrad']
    assert len(idlossops) == 0
    assert len(idlossgradops) == 0
