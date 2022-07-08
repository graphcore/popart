# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
import popart
import test_util as tu
import pytest
import json

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "optimizer_tests"))
import torch_lamb

allOptimizerTypes = ["sgd", "adam", "lamb", "lambnobias"]


def _get_popart_optimizer(optType, clipNormSettings):
    if optType == "sgd":
        return popart.SGD({"defaultLearningRate": (0.1, True)}, clipNormSettings)
    elif optType == "adam":
        return popart.Adam(
            {
                "defaultLearningRate": (0.1, False),
                "defaultBeta1": (0.9, False),
                "defaultBeta2": (0.999, False),
                "lossScaling": (20, False),
            },
            weight_decay_mode=popart.WeightDecayMode.L2Regularization,
            mode=popart.AdamMode.Adam,
            clip_norm_settings=clipNormSettings,
        )
    elif optType == "lamb":
        return popart.Adam(
            {
                "defaultLearningRate": (0.01, False),
                "defaultBeta1": (0.9, False),
                "defaultBeta2": (0.999, False),
                "defaultEps": (1e-06, False),
                "defaultWeightDecay": (0.1, False),
                "lossScaling": (10, False),
            },
            weight_decay_mode=popart.WeightDecayMode.Decay,
            mode=popart.AdamMode.Lamb,
            clip_norm_settings=clipNormSettings,
        )
    elif optType == "lambnobias":
        return popart.Adam(
            {
                "defaultLearningRate": (0.01, False),
                "defaultBeta1": (0.9, False),
                "defaultBeta2": (0.999, False),
                "defaultEps": (1e-06, False),
                "defaultWeightDecay": (0.1, False),
                "lossScaling": (10, True),
            },
            scaled_optimizer_state=False,
            weight_decay_mode=popart.WeightDecayMode.Decay,
            mode=popart.AdamMode.LambNoBias,
            clip_norm_settings=clipNormSettings,
        )
    else:
        raise Exception(f"Unrecognized optimizer type: '{optType}'")


def _get_torch_optimizer(optType, net):
    if optType == "sgd":
        return torch.optim.SGD(net.parameters(), lr=0.1)
    elif optType == "adam":
        return torch.optim.Adam(
            net.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.1
        )
    elif optType == "lamb":
        return torch_lamb.Lamb(
            net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.1
        )
    elif optType == "lambnobias":
        return torch_lamb.Lamb(
            net.parameters(),
            lr=0.01,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.1,
            biasCorrection=False,
        )
    else:
        raise Exception(f"Unrecognized optimizer type: '{optType}'")


# Create a model that performs a series of convs on the
# input, run one training iteration of the model, and return
# the updated weights.
# `clipInfo` describes the gradient clipping groups.
# The format of `clipInfo` is:
#     List(Tuple(List(TensorId), MaxNorm)))
def _run_popart_test_model(
    data,
    weights,
    clipInfo,
    pipelineGroups=None,
    accumulationFactor=None,
    optimizerType=None,
    enablePipelining=False,
):
    # make sure the weights are not accidently modified in this function
    weights = [np.copy(i) for i in weights]
    bld = popart.Builder()
    d0 = bld.addInputTensor(popart.TensorInfo(data))
    # consistently name the weights so we can refer to them later
    weightIds = [
        bld.addInitializedInputTensor(w, f"weight{i}") for i, w in enumerate(weights)
    ]

    # Get a pipeline stage for each weight
    if pipelineGroups:
        pipelineStages = {}
        maxPipelineStage = len(pipelineGroups) - 1
        for pipelineStage, indices in enumerate(pipelineGroups):
            for index in indices:
                pipelineStages[index] = pipelineStage

    x = d0
    for i, weightId in enumerate(weightIds):
        x = bld.aiOnnxOpset9.conv(
            [x, weightId], dilations=[1, 1], pads=[1, 1, 1, 1], strides=[1, 1]
        )
        if pipelineGroups:
            bld.pipelineStage(x, pipelineStages[i])
            bld.virtualGraph(x, pipelineStages[i])

    out = bld.aiGraphcore.l1loss([x], 1.0)
    if pipelineGroups:
        bld.pipelineStage(out, maxPipelineStage)
        bld.virtualGraph(out, maxPipelineStage)

    bld.addOutputTensor(out)

    proto = bld.getModelProto()

    dataFlow = popart.DataFlow(
        1, {x: popart.AnchorReturnType("All"), out: popart.AnchorReturnType("All")}
    )

    if pipelineGroups:
        device = popart.DeviceManager().createIpuModelDevice(
            {"numIPUs": maxPipelineStage + 1}
        )
    else:
        device = popart.DeviceManager().createIpuModelDevice({"numIPUs": 1})

    clipNormSettings = []
    for weightIndices, maxNorm in clipInfo:
        clipNormSettings.append(
            popart.ClipNormSettings.clipWeights(
                [weightIds[i] for i in weightIndices], maxNorm
            )
        )
    opts = popart.SessionOptions()
    opts.enableOutlining = False
    if pipelineGroups:
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.accumulationFactor = accumulationFactor
        opts.enableGradientAccumulation = True
        opts.accumulateOuterFragmentSettings.schedule = (
            popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized
        )
        opts.enablePipelining = enablePipelining

    sess = popart.TrainingSession(
        proto,
        dataFlow=dataFlow,
        loss=out,
        optimizer=_get_popart_optimizer(optimizerType, clipNormSettings),
        deviceInfo=device,
        userOptions=opts,
    )

    sess.prepareDevice()

    sess.weightsFromHost()

    anchors = sess.initAnchorArrays()
    if pipelineGroups:
        data = np.array([data] * accumulationFactor)
    stepio = popart.PyStepIO({d0: data}, anchors)
    sess.run(stepio)

    result = anchors[x]

    sess.weightsToHost()

    resultWeights = {
        weightIds[i]: np.empty(weights[i].shape, dtype=weights[i].dtype)
        for i in range(len(weights))
    }

    weightsio = popart.PyWeightsIO(resultWeights)
    sess.readWeights(weightsio)
    return result, resultWeights


# Create a model that performs a series of convs on the
# input, run one training iteration of the model, and return
# the updated weights.
# `clipInfo` describes the gradient clipping groups.
# The format of `clipInfo` is:
#     List(Tuple(List(TensorId), MaxNorm)))
def _run_torch_test_model(
    data, weights, clipInfo, accumulationFactor=None, optimizerType=None
):
    data = torch.tensor(data)
    weights = [torch.tensor(np.copy(i)) for i in weights]

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.convs = []
            for i in range(len(weights)):
                name = f"conv{i}"
                conv = nn.Conv2d(1, 1, 2, padding=1)
                # Setting convs as a parameter so
                # `net.parameters()` will return the conv
                # weights.
                setattr(self, name, conv)
                self.convs.append(conv)

        def forward(self, x):
            for conv in self.convs:
                x = conv(x)
            return x

    net = Net()
    optimizer = _get_torch_optimizer(optimizerType, net)
    loss = nn.L1Loss()

    for i in range(len(weights)):
        assert net.convs[i].weight.shape == weights[i].shape
        net.convs[i].weight.data = weights[i]
        net.convs[i].bias.data = torch.zeros(net.convs[i].bias.shape)

    optimizer.zero_grad()
    result = net(data)
    output = loss(result, torch.zeros(result.shape))
    output.backward()

    if accumulationFactor:
        for conv in net.convs:
            conv.weight.grad = conv.weight.grad * accumulationFactor

    for weightIndices, maxNorm in clipInfo:
        torch.nn.utils.clip_grad_norm_(
            [net.convs[i].weight for i in weightIndices], maxNorm
        )

    optimizer.step()

    resultWeights = [conv.weight.data.detach().numpy() for conv in net.convs]
    result = result.detach().numpy()

    resultWeights = {
        f"weight{index}": weight for index, weight in enumerate(resultWeights)
    }

    return result, resultWeights


@pytest.mark.parametrize("optimizerType", allOptimizerTypes)
def test_basic(optimizerType):
    np.random.seed(0)
    clipNorm = 0.2

    data = np.random.rand(1, 1, 8, 8).astype(np.float32)
    weights = [np.random.rand(1, 1, 2, 2).astype(np.float32) for _ in range(2)]
    initialWeights = {f"weight{i}": np.copy(w) for i, w in enumerate(weights)}

    popartResult, popartWeights = _run_popart_test_model(
        data, weights, [([0, 1], clipNorm)], optimizerType=optimizerType
    )
    torchResult, torchWeights = _run_torch_test_model(
        data, weights, [([0, 1], clipNorm)], optimizerType=optimizerType
    )

    assert popartResult.shape == torchResult.shape
    assert np.allclose(popartResult, torchResult)

    def print_tensor(x):
        x = str(x)
        x = x.replace("\n", "")
        print(f"  {x}")

    for key in popartWeights.keys():
        print(f"{key}:")
        print_tensor(initialWeights[key])
        print_tensor(popartWeights[key])
        print_tensor(torchWeights[key])

    for key in popartWeights.keys():
        assert np.allclose(popartWeights[key], torchWeights[key], atol=1e-5)


@pytest.mark.parametrize("optimizerType", allOptimizerTypes)
def test_two_groups(optimizerType):
    print()
    np.random.seed(0)
    norm1 = 0.5
    norm2 = 0.2

    data = np.random.rand(1, 1, 8, 8).astype(np.float32)
    weights = [np.random.rand(1, 1, 2, 2).astype(np.float32) for _ in range(4)]
    initialWeights = {f"weight{i}": np.copy(w) for i, w in enumerate(weights)}

    clipGroups = [([0, 1], norm1), ([2, 3], norm2)]

    popartResult, popartWeights = _run_popart_test_model(
        data, weights, clipGroups, optimizerType=optimizerType
    )
    torchResult, torchWeights = _run_torch_test_model(
        data, weights, clipGroups, optimizerType=optimizerType
    )

    assert popartResult.shape == torchResult.shape
    assert np.allclose(popartResult, torchResult)

    def print_tensor(x):
        x = str(x)
        x = x.replace("\n", "")
        print(f"  {x}")

    for key in popartWeights.keys():
        if np.allclose(popartWeights[key], torchWeights[key]):
            print(f"{key}:")
        else:
            print(f"{key}: FAIL")
        print_tensor(initialWeights[key])
        print_tensor(popartWeights[key])
        print_tensor(torchWeights[key])

    for key in popartWeights.keys():
        assert np.allclose(popartWeights[key], torchWeights[key], atol=1e-06)


# piplining is not working for adam yet
@pytest.mark.parametrize("optimizerType", allOptimizerTypes)
@pytest.mark.parametrize("enablePipelining", [True, False])
def test_pipelined(optimizerType, enablePipelining):
    print()
    np.random.seed(0)
    norm1 = 15

    data = np.random.rand(1, 1, 8, 8).astype(np.float32)
    weights = [np.random.rand(1, 1, 2, 2).astype(np.float32) for _ in range(4)]
    initialWeights = {f"weight{i}": np.copy(w) for i, w in enumerate(weights)}

    clipGroups = [([0, 1, 2, 3], norm1)]
    pipelineGroups = ((0, 1), (2, 3))

    popartResult, popartWeights = _run_popart_test_model(
        data,
        weights,
        clipGroups,
        pipelineGroups,
        accumulationFactor=3,
        optimizerType=optimizerType,
        enablePipelining=enablePipelining,
    )
    popartResult = popartResult[1]
    torchResult, torchWeights = _run_torch_test_model(
        data, weights, clipGroups, accumulationFactor=3, optimizerType=optimizerType
    )

    assert popartResult.shape == torchResult.shape
    assert np.allclose(popartResult, torchResult)

    def print_tensor(x):
        x = str(x)
        x = x.replace("\n", "")
        print(f"  {x}")

    for key in popartWeights.keys():
        print(f"{key}:")
        print_tensor(initialWeights[key])
        print_tensor(popartWeights[key])
        print_tensor(torchWeights[key])

    for key in popartWeights.keys():
        assert np.allclose(popartWeights[key], torchWeights[key])


@pytest.mark.parametrize("optimizerType", allOptimizerTypes)
def test_serialized_matmul(optimizerType):
    def run_test(serialization_mode, serialization_factor):
        input_channels = 6
        reducing_dim = 2
        output_channels = 4

        lhs_shape = [input_channels, reducing_dim]
        rhs_shape = [reducing_dim, output_channels]
        lhs_data = np.ones((*lhs_shape,), dtype=np.float32)
        rhs_data = np.ones((*rhs_shape,), dtype=np.float32)

        builder = popart.Builder()

        lhs = builder.addInitializedInputTensor(lhs_data, "lhs")
        rhs = builder.addInitializedInputTensor(rhs_data, "rhs")

        o = builder.aiOnnx.matmul([lhs, rhs])

        builder.setSerializeMatMul({o}, serialization_mode, serialization_factor)

        loss = builder.aiGraphcore.l1loss([o], 0.1)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1,
            {
                o: popart.AnchorReturnType("All"),
                rhs: popart.AnchorReturnType("Final"),
                popart.reservedGradientPrefix() + lhs: popart.AnchorReturnType("All"),
            },
        )

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}
        # TODO(T14786) investigate why swapping causes some tests to fail
        opts.swapLimitScheduler = -1
        # TODO(T14786) investigate why GREEDY causes some tests to fail
        opts.kahnTieBreaker = "FIFO"
        # TODO(T14786) investigate why tighter initialization causes some tests to fail
        opts.transitiveClosureOptimizationThreshold = 0
        # Gradient clipping isn't working for serialized matmuls without gradient accumulation
        opts.enableGradientAccumulation = True

        pat = popart.Patterns(
            ["MatMulOp", "MatMulRhsGradOp", "MatMulLhsGradOp", "OpToIdentity"]
        )
        pat.enableRuntimeAsserts(False)

        optimizer = _get_popart_optimizer(
            optimizerType, [popart.ClipNormSettings([lhs, rhs], 0.01)]
        )

        with tu.create_test_device(opts={"compileIPUCode": False}) as device:
            session = popart.TrainingSession(
                fnModel=proto,
                dataFlow=dataFlow,
                userOptions=opts,
                loss=loss,
                optimizer=optimizer,
                patterns=pat,
                deviceInfo=device,
            )

            session.prepareDevice()

            session.weightsFromHost()

            anchors = session.initAnchorArrays()

            inputs = {lhs: lhs_data}
            stepio = popart.PyStepIO(inputs, anchors)

            session.run(stepio)
            session.weightsToHost()
            print(f"weights should be called {lhs} and {rhs}")

        return anchors

    r1 = run_test("none", 0)
    r2 = run_test("output_channels", 2)

    print(r1)
    print(r2)
    assert len(r1.keys()) > 0
    for key in r1.keys():
        assert np.allclose(r1[key], r2[key])


@pytest.mark.parametrize("optimizerType", allOptimizerTypes)
def test_clipping_all_weights(optimizerType):
    input_channels = 6
    np.random.seed(0)

    a_shape = [input_channels]
    b_shape = [input_channels]
    a_data = np.ones((*a_shape,), dtype=np.float32)
    b_data = np.ones((*b_shape,), dtype=np.float32)
    X_data = np.random.rand(*b_shape).astype(np.float32)
    target_data = np.random.rand(*b_shape).astype(np.float32)

    builder = popart.Builder()

    a = builder.addInitializedInputTensor(a_data, "a")
    b = builder.addInitializedInputTensor(b_data, "b")
    X = builder.addInputTensor(popart.TensorInfo(X_data))
    target = builder.addInputTensor(popart.TensorInfo(target_data))

    # a + b * X
    out = builder.aiOnnx.add([a, builder.aiOnnx.mul([b, X])])
    # MSE loss
    sub = builder.aiOnnx.sub([target, out])
    square = builder.aiOnnx.mul([sub, sub])
    loss = builder.aiGraphcore.identityloss(
        [square], reduction=popart.ReductionType.Mean
    )

    builder.addOutputTensor(out)
    builder.addOutputTensor(loss)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(
        1,
        {out: popart.AnchorReturnType("Final"), loss: popart.AnchorReturnType("Final")},
    )

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}

    optimizer = _get_popart_optimizer(
        optimizerType, [popart.ClipNormSettings.clipAllWeights(0.1)]
    )

    with tu.create_test_device(opts={"compileIPUCode": False}) as device:
        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=dataFlow,
            userOptions=opts,
            loss=loss,
            optimizer=optimizer,
            deviceInfo=device,
        )

        ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))

        # There should be two ReduceSumSquare ops in the graph if gradient clipping
        # was applied to all the weights.
        reduceSumSquareOps = [
            op for op in ir["maingraph"] if op["type"] == "ReduceSumSquare"
        ]
        assert len(reduceSumSquareOps) == 2
