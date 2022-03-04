# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.random as npr

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu_model
def test_multi_loss_pipeline_same():
    run_test_multi_loss_pipeline(same_vgraph=True)


@tu.requires_ipu_model
def test_multi_loss_pipeline_different():
    run_test_multi_loss_pipeline(same_vgraph=False)


def run_test_multi_loss_pipeline(same_vgraph=True):

    seed = 1015
    npr.seed(seed)
    torch.manual_seed(seed)

    # lambda values used in the L1 losses
    lambda1 = 1.0
    lambda2 = 1.0

    defaultLearningRate0 = 0.001
    defaultMomentum0 = 0.38
    defaultDampening0 = 0.5

    # tensor dimensions
    height = 6
    sampleShape = [height, height]
    batchesPerStep = 5
    replicationFactor = 1
    accumulationFactor = 4
    nVirtualGraphs = 2
    samplesPerBatch = 48
    divvyFactor = replicationFactor * accumulationFactor
    if (samplesPerBatch % divvyFactor != 0):
        raise RuntimeError(
            "Invalid divvy factor, does not divide samplesPerBatch")
    samplesPerMicroBatch = samplesPerBatch // divvyFactor
    nIPUs = replicationFactor * nVirtualGraphs
    stepDataShape = [batchesPerStep, samplesPerBatch, height, height]
    microBatchShape = [samplesPerMicroBatch, height, height]
    stepDataInfo = popart.TensorInfo("FLOAT", stepDataShape)
    microBatchInfo = popart.TensorInfo("FLOAT", microBatchShape)

    #initial weight and input values
    w0vals = np.array(npr.randn(height, height), dtype=np.float32)
    w1vals = np.array(npr.randn(height, height), dtype=np.float32)
    inputVals = np.array(npr.randn(*stepDataShape), dtype=np.float32)

    # Build the ONNX Model
    builder = popart.Builder()
    input0 = builder.addInputTensor(microBatchInfo)
    w0 = builder.addInitializedInputTensor(w0vals)
    w1 = builder.addInitializedInputTensor(w1vals)
    scaleFactor = 1. / np.sqrt(height + 0.)

    # all compute on IPU 0.
    with builder.virtualGraph(0):
        mm0 = builder.aiOnnx.matmul([input0, w0])
        mm1 = builder.aiOnnx.matmul([mm0, w1])
        scale1 = builder.aiGraphcore.scale([mm1], scaleFactor)
        skipOut = builder.aiOnnx.add([mm0, scale1])

    with builder.virtualGraph(1 if same_vgraph else 0):
        loss2 = builder.aiGraphcore.l1loss([skipOut],
                                           lambda2,
                                           reduction=popart.ReductionType.Sum)

    with builder.virtualGraph(1):
        loss1 = builder.aiGraphcore.l1loss([scale1],
                                           lambda1,
                                           reduction=popart.ReductionType.Sum)
        finalLoss = builder.aiOnnx.sum([loss1, loss2])

    # input0  w0
    #    |    |
    #    matmul    w1
    #      |  |    /
    #      |  matmul
    #      |  |
    #      | scale
    #      |  | |
    #      add  |
    #       |   |
    # - - - - - - - - - - -|
    #       |   |          |
    #     loss  |          |
    #======================| <-- if same_vgraph == False
    #           |          |
    #         loss         |
    # - - - - - - - - - - -|
    #       |   |          |
    #======================| <-- if same_vgraph == True
    #       |   |          |
    #     loss loss        |
    # - - - - - - - - - - -|
    #

    anchors = {}
    dataFlow = popart.DataFlow(batchesPerStep, anchors)

    def getWeights(withPipelining):

        with tu.create_test_device(numIpus=nIPUs, tilesPerIPU=1216) as device:
            userOptions = popart.SessionOptions()
            userOptions.enableOutlining = False
            userOptions.enablePipelining = withPipelining
            userOptions.enableGradientAccumulation = True
            userOptions.accumulationFactor = accumulationFactor
            userOptions.virtualGraphMode = popart.VirtualGraphMode.Manual

            session = popart.TrainingSession(
                fnModel=builder.getModelProto(),
                dataFlow=dataFlow,
                optimizer=popart.SGD({
                    "defaultLearningRate": (defaultLearningRate0, False),
                    "defaultMomentum": (defaultMomentum0, False),
                    "defaultDampening": (defaultDampening0, False)
                }),
                loss=finalLoss,
                userOptions=userOptions,
                deviceInfo=device)

            anchorArrays = session.initAnchorArrays()

            session.prepareDevice()
            session.weightsFromHost()

            stepio = popart.PyStepIO({input0: inputVals}, anchorArrays)
            session.run(stepio)
            session.weightsToHost()
            w0R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
            w1R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
            weightsRead = popart.PyWeightsIO({w0: w0R, w1: w1R})
            session.readWeights(weightsRead)
            return w0R, w1R

    # pytorch verification model:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.w0 = torch.nn.Parameter(torch.from_numpy(w0vals.copy()))
            self.w1 = torch.nn.Parameter(torch.from_numpy(w1vals.copy()))

        def forward(self, x, _):  # i is an unused parameter needed in torch

            mm0 = torch.matmul(x, self.w0)
            mm1 = torch.matmul(mm0, self.w1)
            scl1 = mm1 * scaleFactor
            skip = scl1 + mm0
            return scl1, skip

    net = Net()

    optimizer = optim.SGD(net.parameters(),
                          lr=defaultLearningRate0,
                          momentum=defaultMomentum0,
                          dampening=defaultDampening0)
    for i in range(batchesPerStep):
        dr1, dr2 = net(torch.from_numpy(inputVals[i]), i)
        loss = torch.sum(lambda1 * torch.abs(dr1) + lambda2 * torch.abs(dr2))
        # caveat on the SGD see TODO T13098
        if (i == 0):
            loss *= (1 - defaultDampening0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w0R_with, w1R_with = getWeights(True)
    w0R_out, w1R_out = getWeights(False)

    delta0 = np.sum(np.abs(net.w0.detach().numpy() - w0vals))
    delta1 = np.sum(np.abs(net.w1.detach().numpy() - w1vals))
    print("pytorch baseline")
    print("Total moved by w0: ", delta0)
    print("Total moved by w1: ", delta1)

    error0 = np.sum(np.abs(w0R_out - net.w0.detach().numpy())) / delta0
    error1 = np.sum(np.abs(w1R_out - net.w1.detach().numpy())) / delta1
    print("without pipelining")
    print("Total moved by w0: ", np.sum(np.abs(w0R_out - w0vals)))
    print("Total moved by w1: ", np.sum(np.abs(w1R_out - w1vals)))
    print("l1 error for w0: ", error0)
    print("l1 error for w1: ", error1)
    assert (error0 < 1e-5)
    assert (error1 < 1e-5)

    error0 = np.sum(np.abs(w0R_with - net.w0.detach().numpy())) / delta0
    error1 = np.sum(np.abs(w1R_with - net.w1.detach().numpy())) / delta1
    print("with pipelining")
    print("Total moved by w0: ", np.sum(np.abs(w0R_with - w0vals)))
    print("Total moved by w1: ", np.sum(np.abs(w1R_with - w1vals)))
    print("l1 error for w0: ", error0)
    print("l1 error for w1: ", error1)
    assert (error0 < 1e-5)
    assert (error1 < 1e-5)
