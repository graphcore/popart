# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import test_util as tu

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import numpy.random as npr


def test_manual_serialization():

    # Basic model:
    #
    #  X: data input if shape (N, C0)
    #  W: weight input of shape (C0, C1)
    #
    #  Y    = matmul(X, W)
    #  Z    = relu(Y)
    #  loss = l1Loss(Z)
    #
    # With array dimensions

    N = 12
    C0 = 244
    C1 = 286

    # In this test, we manually serialise the matmul, converting
    # matmul ((N,C0) , (C0,C1))
    #
    # into a sequence of factor-f smaller matmuls
    # matmul (N,C0/f),(C0/f,C1))
    #
    # reapeated and accumulated f times, where f is

    f = 4
    assert (C0 % f == 0)

    # Constructing the model

    builder = popart.Builder()
    wVals = np.array(npr.randn(C0, C1), dtype=np.float32)
    W = builder.addInitializedInputTensor(wVals)
    xInfo = popart.TensorInfo("FLOAT", [N, C0])
    X = builder.addInputTensor(xInfo)
    axesV = np.array([0, 1]).astype(np.int32)
    axes = builder.addInitializedInputTensor(axesV)

    for i in range(f):
        # the lower index of the i'th slice
        lwr = int(i * C0 / f)

        # the upper index of the i'th slice
        upp = int((i + 1) * C0 / f)

        # Take a slice of size (N,C0/f) out of X
        s0 = builder.addInitializedInputTensor(
            np.array([0, lwr]).astype(np.int32))
        e0 = builder.addInitializedInputTensor(
            np.array([N, upp]).astype(np.int32))
        X_slice = builder.aiOnnx.slice([X, s0, e0, axes])

        # Take a slice of size (C0/f,C1) out of W
        s1 = builder.addInitializedInputTensor(
            np.array([lwr, 0]).astype(np.int32))
        e1 = builder.addInitializedInputTensor(
            np.array([upp, C1]).astype(np.int32))
        W_slice = builder.aiOnnx.slice([W, s1, e1, axes])

        # Multiply the slices together, and accumulate as necessary
        mm_part = builder.aiOnnx.matmul([X_slice, W_slice])
        if i == 0:
            Y = mm_part

        else:
            Y = builder.aiOnnx.add([mm_part, Y])

    # Finally, the non-linearity
    Z = builder.aiOnnx.relu([Y])

    # This boiler-plate is currently necessary with opset-10 slice
    graph_transformer = popart.GraphTransformer(builder.getModelProto())
    graph_transformer.convertAllFixedPointInitializersToConstants()
    builder = popart.Builder(graph_transformer.getModelProto())

    loss1 = popart.L1Loss(Z, "l1LossVal1", 0.2)
    dataFlow = popart.DataFlow(1, {})
    device = tu.create_test_device()
    userOptions = popart.SessionOptions()

    # To obtain the final dot graph, uncomment this:
    # userOptions.dotChecks = {popart.DotCheck.FINAL}

    patterns = popart.Patterns()

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFeed=dataFlow,
                                     optimizer=popart.SGD(
                                         {"defaultLearningRate": (0.1, True)}),
                                     losses=[loss1],
                                     passes=patterns,
                                     userOptions=userOptions,
                                     deviceInfo=device)

    session.prepareDevice()
    session.weightsFromHost()

    inputVals = np.array(npr.randn(1 * N * C0), dtype=np.float32)
    stepio = popart.PyStepIO({X: inputVals}, {})
    session.run(stepio)
    session.weightsToHost()
    w0R = np.array(-777.0 * np.ones(C0 * C1), dtype=np.float32)
    weightsRead = popart.PyWeightsIO({W: w0R})
    session.readWeights(weightsRead)

    # A pytorch version to confirm numerical correctness:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.w0 = torch.nn.Parameter(torch.from_numpy(wVals.copy()))

        def forward(self, x):
            return torch.relu(torch.matmul(x, self.w0))

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    out = net(torch.from_numpy(inputVals.reshape([N, C0])))
    loss = 0.2 * torch.sum(torch.abs(out))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    baseline0 = np.sum(
        np.abs(net.w0.detach().numpy().flatten() - wVals.flatten()))
    baseline1 = np.sum(np.abs(w0R - wVals.flatten()))
    error = np.sum(np.abs(np.abs(net.w0.detach().numpy().flatten() - w0R)))

    assert (error / (baseline0 + baseline1) < 1e-6)
