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


def test_against_pytorch():
    """
    Comparison of popart and PyTorch optimizers, and the changes needed to PyTorch 
    to match popart. Note that these differences should have no effect on overall 
    training measures, and this discussion is just for those interested in exact 
    reproducility between popart and pytorch 

    The main differences are:
    1)
    pytorch optimizer, in the very first iteration with a new optimizer: 
    initializes velocity tensor with zeros and does not do damping,
    popart optimizer, in the first iteration with a new optimizer:
    retains velocity tensor from previous state with previous optimizer, 
    or sets it to zero if the first round of training

    2)
    popart and pytorch updates the optimizer at a different "phase" 
    a)    v <- v * mm  + (1 - dp) * wd * w
    b)    v <- v + (1 - dp) * g
    c)    w <- w - lr * v

    pytorch goes (abc)(abc)(abc)(abc) 
    popart goes  (abca)(bca)(bca)(bca)

    where changes to the optimizer can be done between periods. For this reason, 
    updates to mm, dp, and wd have different effects. 

    See also sgd_mixed_mode_test_cpp_1_3.cpp
    """

    #optimizer parameters
    defaultLearningRate0 = 0.005
    defaultLearningRate1 = 0.003
    defaultLearningRate2 = 0.001

    defaultMomentum0 = 0.1
    defaultDampening0 = 0.3
    lossScaling0 = 10.0
    defaultVelocityScaling0 = 0.5
    defaultWeightDecay0 = 0.01

    optMap0 = {
        "defaultLearningRate": (defaultLearningRate0, False),
        "defaultMomentum": (defaultMomentum0, False),
        "defaultDampening": (defaultDampening0, False),
        "defaultVelocityScaling": (defaultVelocityScaling0, False),
        "lossScaling": (lossScaling0, False),
        "defaultWeightDecay": (defaultWeightDecay0, False)
    }

    optMap1 = {
        "defaultLearningRate": (defaultLearningRate1, False),
        "defaultMomentum": (defaultMomentum0, False),
        "defaultDampening": (defaultDampening0, False),
        "defaultVelocityScaling": (defaultVelocityScaling0, False),
        "lossScaling": (lossScaling0, False),
        "defaultWeightDecay": (defaultWeightDecay0, False)
    }

    optMap2 = {
        "defaultLearningRate": (defaultLearningRate2, False),
        "defaultMomentum": (defaultMomentum0, False),
        "defaultDampening": (defaultDampening0, False),
        "defaultVelocityScaling": (defaultVelocityScaling0, False),
        "lossScaling": (lossScaling0, False),
        "defaultWeightDecay": (defaultWeightDecay0, False)
    }
    seed = 1015
    npr.seed(seed)
    torch.manual_seed(seed)

    #L1 loss value
    lambda1 = 1.0

    # tensor dimensions and replications
    height = 2
    numberOfSteps = 3
    batchesPerStep = 5
    sampleShape = [height, height]
    replicationFactor = 1
    accumulationFactor = 1
    nVirtualGraphs = 1
    samplesPerBatch = 4
    divvyFactor = replicationFactor * accumulationFactor
    samplesPerMicroBatch = samplesPerBatch // divvyFactor
    nIPUs = replicationFactor * nVirtualGraphs
    stepDataShape = [batchesPerStep, samplesPerBatch, height, height]
    microBatchShape = [samplesPerMicroBatch, height, height]
    stepDataInfo = popart.TensorInfo("FLOAT", stepDataShape)
    microBatchInfo = popart.TensorInfo("FLOAT", microBatchShape)

    #initial weight and input values
    w0vals = np.array(npr.randn(height, height), dtype=np.float32)
    w1vals = np.array(npr.randn(height, height), dtype=np.float32)
    inputVals = [
        np.array(npr.randn(*stepDataShape), dtype=np.float32)
        for i in range(numberOfSteps)
    ]

    # Build the ONNX Model
    builder = popart.Builder()
    input0 = builder.addInputTensor(microBatchInfo)
    w0 = builder.addInitializedInputTensor(w0vals)
    w1 = builder.addInitializedInputTensor(w1vals)

    # Model:
    #
    # input  w0     w1
    #     \  |      |
    #      mul  -  add - L1 loss

    mm0 = builder.aiOnnx.mul([input0, w0])
    mm1 = builder.aiOnnx.add([mm0, w1])
    builder.addOutputTensor(mm1)
    loss1 = popart.L1Loss(mm1, "l1LossVal1", lambda1)
    dataFlow = popart.DataFlow(batchesPerStep, {})
    options = {'numIPUs': nIPUs, "tilesPerIPU": 1216}
    device = popart.DeviceManager().createIpuModelDevice(options)
    userOptions = popart.SessionOptions()
    userOptions.enableGradientAccumulation = False
    userOptions.enablePrefetchDatastreams = False

    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFeed=dataFlow,
        userOptions=userOptions,
        losses=[loss1],
        optimizer=popart.SGD(optMap0),
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    anchorArrays = session.initAnchorArrays()

    session.prepareDevice()
    session.weightsFromHost()

    session.optimizerFromHost()
    stepio = popart.PyStepIO({input0: inputVals[0]}, anchorArrays)
    session.run(stepio)

    session.updateOptimizer(popart.SGD(optMap1))
    session.optimizerFromHost()
    stepio = popart.PyStepIO({input0: inputVals[1]}, anchorArrays)
    session.run(stepio)

    session.updateOptimizer(popart.SGD(optMap2))
    session.optimizerFromHost()
    stepio = popart.PyStepIO({input0: inputVals[2]}, anchorArrays)
    session.run(stepio)

    session.weightsToHost()
    w0R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
    w1R = np.array(-777.0 * np.ones(sampleShape), dtype=np.float32)
    weightsRead = popart.PyWeightsIO({w0: w0R, w1: w1R})
    session.readWeights(weightsRead)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.w0 = torch.nn.Parameter(torch.from_numpy(w0vals.copy()))
            self.w1 = torch.nn.Parameter(torch.from_numpy(w1vals.copy()))

        def forward(self, x, i):
            mm0 = torch.mul(x, self.w0)
            mm1 = torch.add(mm0, self.w1)
            return mm1

    net = Net()

    optMaps = [optMap0, optMap1, optMap2]

    for step in range(3):
        if step is 0:
            oldOptimizer = None
        else:
            oldOptimizer = optimizer

        optimizer = optim.SGD(
            net.parameters(),
            lr=optMaps[step]["defaultLearningRate"][0],
            momentum=optMaps[step]["defaultMomentum"][0],
            dampening=optMaps[step]["defaultDampening"][0],
            weight_decay=optMaps[step]["defaultWeightDecay"][0])

        if step is 0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    param_state = optimizer.state[p][
                        'momentum_buffer'] = p.data * 0

        else:
            for group, oldGroup in zip(optimizer.param_groups,
                                       oldOptimizer.param_groups):
                for p, oldp in zip(group['params'], oldGroup['params']):
                    param_state = optimizer.state[p][
                        'momentum_buffer'] = oldOptimizer.state[p][
                            'momentum_buffer']

        for i in range(batchesPerStep):
            out = net(torch.from_numpy(inputVals[step][i]), i)
            loss = lambda1 * torch.sum(torch.abs(out))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    delta0 = np.sum(np.abs(net.w0.detach().numpy() - w0vals))
    delta1 = np.sum(np.abs(net.w1.detach().numpy() - w1vals))
    print("pytorch baseline")
    print("Total moved by w0: ", delta0)
    print("Total moved by w1: ", delta1)

    error0 = np.sum(np.abs(w0R - net.w0.detach().numpy())) / delta0
    error1 = np.sum(np.abs(w1R - net.w1.detach().numpy())) / delta1
    print("without pipelining")
    print("Total moved by w0: ", np.sum(np.abs(w0R - w0vals)))
    print("Total moved by w1: ", np.sum(np.abs(w1R - w1vals)))
    print("l1 error for w0: ", error0)
    print("l1 error for w1: ", error1)
    assert (error0 < 1e-5)
    assert (error1 < 1e-5)
