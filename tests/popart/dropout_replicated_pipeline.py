# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy.random as npr

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

replicationFactor = 2
nVirtualGraphs = 2
nIPUs = replicationFactor * nVirtualGraphs


def runTest(forceAddOutOfPlace, pipelineRecomputation):
    """
    Test of pipelining with dropout, recomputation, graph replication, 
    gradient accumulation
    """
    #Has dependencies on T12562. T12976, T13098 for full support

    seed = 1015
    npr.seed(seed)
    torch.manual_seed(seed)

    #L1 loss value
    lambda1 = 1.0

    #optimizer parameters
    defaultLearningRate0 = 0.001
    defaultMomentum0 = 0.01
    defaultDampening0 = 0.5
    lossScaling0 = 10.0
    defaultVelocityScaling0 = 0.15
    defaultWeightDecay0 = 0.01

    # tensor dimensions and replications
    height = 6
    batchesPerStep = 5
    sampleShape = [height, height]
    accumulationFactor = 4
    samplesPerBatch = 48
    divvyFactor = replicationFactor * accumulationFactor
    if (samplesPerBatch % divvyFactor != 0):
        raise RuntimeError("Invalid divvy factor")
    samplesPerMicroBatch = samplesPerBatch // divvyFactor
    stepDataShape = [batchesPerStep, samplesPerBatch, height, height]
    microBatchShape = [samplesPerMicroBatch, height, height]
    stepDataInfo = popart.TensorInfo("FLOAT", stepDataShape)
    microBatchInfo = popart.TensorInfo("FLOAT", microBatchShape)

    #initial weight and input values
    w0vals = np.array(npr.randn(height, height), dtype=np.float32)
    w1vals = np.array(npr.randn(height, height), dtype=np.float32)
    w2vals = np.array(npr.randn(height, height), dtype=np.float32)
    inputVals = np.array(npr.randn(*stepDataShape), dtype=np.float32)

    # Build the ONNX Model
    builder = popart.Builder()
    input0 = builder.addInputTensor(microBatchInfo)
    w0 = builder.addInitializedInputTensor(w0vals)
    w1 = builder.addInitializedInputTensor(w1vals)
    w2 = builder.addInitializedInputTensor(w2vals)

    scaleFactor = 1. / np.sqrt(height + 0.)

    # Model:
    #
    # input  w0                            w1
    #     \  |                            /
    #     matmul - scale -> dropout -> matul
    #         \                        |
    #         |                       scale
    #         |                        |
    #         |                      dropout
    #         |                        /\
    #       add  -------<---<----<----   \
    #        |                            |
    #     dropout                     scale by 2
    #        |                            |
    #  = = = | = = = = = IPU barrier = = =|= = = = = =
    #        |                            |
    #        |   w2                       |
    #        |  /                         |
    #       matmul                       /
    #        |                          /
    #      scale                       /
    #        |                        /
    #      dropout                   /
    #        |                       |
    #        ------->---->---->---> add -> L1 loss (lambda 2)

    with builder.virtualGraph(0):
        mm0 = builder.aiOnnx.matmul([input0, w0])
        scale0 = builder.aiGraphcore.scale([mm0], scaleFactor)
        ratio0 = 0.35
        [dropout0, mask0] = builder.aiOnnx.dropout([scale0],
                                                   num_outputs=2,
                                                   ratio=ratio0)
        mm1 = builder.aiOnnx.matmul([dropout0, w1])
        scale1 = builder.aiGraphcore.scale([mm1], scaleFactor)
        ratio1 = 0.5
        [dropout1, mask1] = builder.aiOnnx.dropout([scale1],
                                                   num_outputs=2,
                                                   ratio=ratio1)
        dropout1 = builder.aiGraphcore.scale([dropout1], 2.0)
        skipOut = builder.aiOnnx.add([mm0, dropout1])
        # See resolved task T13137
        if forceAddOutOfPlace:
            builder.setInplacePreferences(skipOut, {"AddRhsInplace": -1.0})

        ratioSkip = 0.6
        [dropoutSkip, maskSkip] = builder.aiOnnx.dropout([skipOut],
                                                         num_outputs=2,
                                                         ratio=ratioSkip)

        # see T13142: we do this so that the recomputation does not modify the anchors
        mask0 = builder.aiOnnx.identity([mask0])
        mask1 = builder.aiOnnx.identity([mask1])
        maskSkip = builder.aiOnnx.identity([maskSkip])

    with builder.virtualGraph(1):
        mm2 = builder.aiOnnx.matmul([dropoutSkip, w2])
        scale2 = builder.aiGraphcore.scale([mm2], scaleFactor)
        ratio2 = 0.7
        [dropout2, mask2] = builder.aiOnnx.dropout([scale2],
                                                   num_outputs=2,
                                                   ratio=ratio2)

        out = builder.aiOnnx.add([dropout2, dropout1])

        # see T13142: we do this so that the recomputation does not modify the anchors
        mask2 = builder.aiOnnx.identity([mask2])

    builder.addOutputTensor(out)

    loss1 = popart.L1Loss(out, "l1LossVal1", lambda1)
    loss1.virtualGraph(1)

    anchors = {
        mask0: popart.AnchorReturnType("All"),
        mask1: popart.AnchorReturnType("All"),
        mask2: popart.AnchorReturnType("All"),
        maskSkip: popart.AnchorReturnType("All"),
    }

    dataFlow = popart.DataFlow(batchesPerStep, anchors)

    device = tu.create_test_device(numIpus=nIPUs)
    assert device

    userOptions = popart.SessionOptions()
    # This requires T12562 to be solved before enabling (TODO)
    userOptions.enableOutlining = False
    userOptions.enablePipelining = True
    userOptions.enableGradientAccumulation = True
    userOptions.accumulationFactor = accumulationFactor

    if pipelineRecomputation:
        userOptions.autoRecomputation = popart.RecomputationType.Pipeline

    if (replicationFactor > 1):
        userOptions.enableReplicatedGraphs = True
        userOptions.replicatedGraphCount = replicationFactor
    userOptions.virtualGraphMode = popart.VirtualGraphMode.Manual

    # TODO https://phabricator.sourcevertex.net/T14035
    userOptions.enablePrefetchDatastreams = False
    #  passes:
    userOptions.engineOptions = {"exchange.streamBufferOverlap": "any"}
    #  fails:
    #  userOptions.engineOptions = {"exchange.streamBufferOverlap" : "hostRearrangeOnly"}

    patterns = popart.Patterns()
    patterns.InPlace = True

    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFeed=dataFlow,
        optimizer=popart.SGD({
            "defaultLearningRate": (defaultLearningRate0, False),
            "defaultMomentum": (defaultMomentum0, False),
            "defaultDampening": (defaultDampening0, False),
            "defaultVelocityScaling": (defaultVelocityScaling0, False),
            "lossScaling": (lossScaling0, True),
            "defaultWeightDecay": (defaultWeightDecay0, True)
        }),
        losses=[loss1],
        passes=patterns,
        userOptions=userOptions,
        deviceInfo=device)

    anchorArrays = session.initAnchorArrays()

    session.prepareDevice()
    session.setRandomSeed(7)
    session.weightsFromHost()

    stepio = popart.PyStepIO({input0: inputVals}, anchorArrays)
    session.run(stepio)
    session.weightsToHost()
    w0R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
    w1R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
    w2R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
    weightsRead = popart.PyWeightsIO({w0: w0R, w1: w1R, w2: w2R})
    session.readWeights(weightsRead)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # merge replication, accumulation
            flattenedShape = [anchorArrays[mask0].shape[0], -1, height, height]
            self.w0 = torch.nn.Parameter(torch.from_numpy(w0vals.copy()))
            self.mask0 = torch.from_numpy(
                anchorArrays[mask0].reshape(flattenedShape))

            self.w1 = torch.nn.Parameter(torch.from_numpy(w1vals.copy()))
            self.mask1 = torch.from_numpy(
                anchorArrays[mask1].reshape(flattenedShape))

            self.maskSkip = torch.from_numpy(
                anchorArrays[maskSkip].reshape(flattenedShape))

            self.w2 = torch.nn.Parameter(torch.from_numpy(w2vals.copy()))
            self.mask2 = torch.from_numpy(
                anchorArrays[mask2].reshape(flattenedShape))

        def forward(self, x, i):
            mm0 = torch.matmul(x, self.w0)
            dr0 = mm0 * scaleFactor * self.mask0[i].type(
                torch.FloatTensor) / (1 - ratio0)

            mm1 = torch.matmul(dr0, self.w1)
            dr1 = mm1 * scaleFactor * self.mask1[i].type(
                torch.FloatTensor) / (1 - ratio1)
            dr1 = 2 * dr1

            drSkip = (dr1 + mm0) * self.maskSkip[i].type(
                torch.FloatTensor) / (1 - ratioSkip)

            mm2 = torch.matmul(drSkip, self.w2)
            dr2 = mm2 * scaleFactor * self.mask2[i].type(
                torch.FloatTensor) / (1 - ratio2)

            out = dr1 + dr2
            return out

    net = Net()

    optimizer = optim.SGD(net.parameters(),
                          lr=defaultLearningRate0,
                          momentum=defaultMomentum0,
                          dampening=defaultDampening0,
                          weight_decay=defaultWeightDecay0)

    # caveat : alternative work-around for TODO T13098
    for group in optimizer.param_groups:
        for p in group['params']:
            param_state = optimizer.state[p]
            param_state['momentum_buffer'] = p.data * 0

    for i in range(batchesPerStep):
        out = net(torch.from_numpy(inputVals[i]), i)
        loss = lambda1 * torch.sum(torch.abs(out))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    delta0 = np.sum(np.abs(net.w0.detach().numpy() - w0vals))
    delta1 = np.sum(np.abs(net.w1.detach().numpy() - w1vals))
    delta2 = np.sum(np.abs(net.w2.detach().numpy() - w2vals))
    print("pytorch baseline")
    print("Total moved by w0: ", delta0)
    print("Total moved by w1: ", delta1)
    print("Total moved by w2: ", delta2)

    error0 = np.sum(np.abs(w0R - net.w0.detach().numpy())) / delta0
    error1 = np.sum(np.abs(w1R - net.w1.detach().numpy())) / delta1
    error2 = np.sum(np.abs(w2R - net.w2.detach().numpy())) / delta2
    print("without pipelining")
    print("Total moved by w0: ", np.sum(np.abs(w0R - w0vals)))
    print("Total moved by w1: ", np.sum(np.abs(w1R - w1vals)))
    print("Total moved by w2: ", np.sum(np.abs(w2R - w2vals)))
    print("l1 error for w0: ", error0)
    print("l1 error for w1: ", error1)
    print("l1 error for w2: ", error2)
    assert (error0 < 1e-5)
    assert (error1 < 1e-5)
    assert (error2 < 1e-5)


@tu.requires_ipu
def test_all_cases():
    # this unit test checks a previously failing case
    runTest(forceAddOutOfPlace=False, pipelineRecomputation=False)

    # with all features on,
    runTest(forceAddOutOfPlace=False, pipelineRecomputation=True)

    print("test_all_cases complete")
