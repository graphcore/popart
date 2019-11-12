import popart
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy.random as npr


def test_multi_loss_pipeline(tmpdir):

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

    # all compute on IPU 0, all losses on IPU 1.
    with builder.virtualGraph(0):
        mm0 = builder.aiOnnx.matmul([input0, w0])
        mm1 = builder.aiOnnx.matmul([mm0, w1])
        scale1 = builder.aiGraphcore.scale([mm1], scaleFactor)
        skipOut = builder.aiOnnx.add([mm0, scale1])

    builder.addOutputTensor(scale1)
    builder.addOutputTensor(skipOut)

    loss1 = popart.L1Loss(scale1, "l1LossVal1", lambda1)
    loss2 = popart.L1Loss(skipOut, "l1LossVal2", lambda2)

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
    #======================
    #       |   |
    #     loss loss
    #
    #
    #
    #

    loss1.virtualGraph(1)
    loss2.virtualGraph(1)

    anchors = {}
    dataFlow = popart.DataFlow(batchesPerStep, anchors)

    def getWeights(withPipelining):

        options = {'numIPUs': nIPUs, "tilesPerIPU": 1216}
        deviceManager = popart.DeviceManager()
        device = deviceManager.createIpuModelDevice(options)
        userOptions = popart.SessionOptions()
        userOptions.enableOutlining = False
        userOptions.enablePipelining = withPipelining
        userOptions.enableGradientAccumulation = True
        userOptions.accumulationFactor = accumulationFactor
        userOptions.virtualGraphMode = popart.VirtualGraphMode.Manual

        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=dataFlow,
            optimizer=popart.SGD({
                "defaultLearningRate": (defaultLearningRate0, False),
                "defaultMomentum": (defaultMomentum0, False),
                "defaultDampening": (defaultDampening0, False)
            }),
            losses=[loss1, loss2],
            userOptions=userOptions,
            deviceInfo=device)

        anchorArrays = session.initAnchorArrays()

        session.prepareDevice()
        session.weightsFromHost()
        session.optimizerFromHost()
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

        def forward(self, x, i):

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
