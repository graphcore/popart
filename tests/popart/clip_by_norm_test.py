# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import popart
import test_util as tu


# Create a model that performs a series of convs on the
# input, run one training iteration of the model, and return
# the updated weights.
# `clipInfo` describes the gradient clipping groups.
# The format of `clipInfo` is:
#     List(Tuple(List(TensorId), MaxNorm)))
def _run_popart_test_model(data, weights, clipInfo):
    # make sure the weights are not accidently modified in this function
    weights = [np.copy(i) for i in weights]
    bld = popart.Builder()
    d0 = bld.addInputTensor(popart.TensorInfo(data))
    # consistently name the weights so we can refer to them later
    weightIds = [
        bld.addInitializedInputTensor(w, f'weight{i}')
        for i, w in enumerate(weights)
    ]

    x = d0
    for weightId in weightIds:
        x = bld.aiOnnxOpset9.conv([x, weightId],
                                  dilations=[1, 1],
                                  pads=[1, 1, 1, 1],
                                  strides=[1, 1])

    out = bld.aiGraphcore.l1loss([x], 1.0)

    bld.addOutputTensor(out)

    proto = bld.getModelProto()

    data_flow = popart.DataFlow(1, {
        x: popart.AnchorReturnType("All"),
        out: popart.AnchorReturnType("All")
    })

    device = popart.DeviceManager().createIpuModelDevice({"numIPUs": 1})

    clipNormSettings = []
    for weightIndices, maxNorm in clipInfo:
        clipNormSettings.append(
            popart.ClipNormSettings([weightIds[i] for i in weightIndices],
                                    maxNorm))
    opts = popart.SessionOptions()

    sess = popart.TrainingSession(
        proto,
        dataFlow=data_flow,
        loss=out,
        # optimizer=popart.ConstSGD(0.1, clip_norm_settings=clipNormSettings),
        optimizer=popart.SGD({"defaultLearningRate": (0.1, True)},
                             clipNormSettings),
        deviceInfo=device,
        userOptions=opts)

    sess.prepareDevice()

    sess.weightsFromHost()

    anchors = sess.initAnchorArrays()
    stepio = popart.PyStepIO({d0: data}, anchors)
    sess.run(stepio)

    result = anchors[x]

    # print(f'Result: {result}')
    # print(f'Loss: {anchors[out]}')

    sess.weightsToHost()

    result_weights = {
        weightIds[i]: np.empty(weights[i].shape, dtype=weights[i].dtype)
        for i in range(len(weights))
    }

    weightsio = popart.PyWeightsIO(result_weights)
    sess.readWeights(weightsio)
    return result, result_weights


# Create a model that performs a series of convs on the
# input, run one training iteration of the model, and return
# the updated weights.
# `clipInfo` describes the gradient clipping groups.
# The format of `clipInfo` is:
#     List(Tuple(List(TensorId), MaxNorm)))
def _run_torch_test_model(data, weights, clipInfo):
    data = torch.tensor(data)
    weights = [torch.tensor(np.copy(i)) for i in weights]

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.convs = []
            for i, w in enumerate(weights):
                name = f'conv{i}'
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
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss = nn.L1Loss()

    for i in range(len(weights)):
        assert net.convs[i].weight.shape == weights[i].shape
        net.convs[i].weight.data = weights[i]
        net.convs[i].bias.data = torch.zeros(net.convs[i].bias.shape)

    optimizer.zero_grad()
    result = net(data)
    output = loss(result, torch.zeros(result.shape))
    output.backward()
    for weightIndices, maxNorm in clipInfo:
        torch.nn.utils.clip_grad_norm_(
            [net.convs[i].weight for i in weightIndices], maxNorm)
    optimizer.step()

    result_weights = [conv.weight.data.detach().numpy() for conv in net.convs]
    result = result.detach().numpy()

    result_weights = {
        f'weight{index}': weight
        for index, weight in enumerate(result_weights)
    }

    # print(f'Result: {result}')
    # print(f'Loss: {output}')

    return result, result_weights


def test_basic():
    print()
    np.random.seed(0)
    clip_norm = 0.2

    data = np.random.rand(1, 1, 8, 8).astype(np.float32)
    weights = [np.random.rand(1, 1, 2, 2).astype(np.float32) for _ in range(2)]
    initial_weights = {f'weight{i}': np.copy(w) for i, w in enumerate(weights)}

    popart_result, popart_weights = _run_popart_test_model(
        data, weights, [([0, 1], clip_norm)])
    torch_result, torch_weights = _run_torch_test_model(
        data, weights, [([0, 1], clip_norm)])

    assert popart_result.shape == torch_result.shape
    assert np.allclose(popart_result, torch_result)

    def print_tensor(x):
        x = str(x)
        x = x.replace('\n', '')
        print(f'  {x}')

    for key in popart_weights.keys():
        print(f'{key}:')
        print_tensor(initial_weights[key])
        print_tensor(popart_weights[key])
        print_tensor(torch_weights[key])

    for key in popart_weights.keys():
        assert np.allclose(popart_weights[key], torch_weights[key])


def test_two_groups():
    print()
    np.random.seed(0)
    norm1 = 0.5
    norm2 = 0.2

    data = np.random.rand(1, 1, 8, 8).astype(np.float32)
    weights = [np.random.rand(1, 1, 2, 2).astype(np.float32) for _ in range(4)]
    initial_weights = {f'weight{i}': np.copy(w) for i, w in enumerate(weights)}

    clipGroups = [([0, 1], norm1), ([2, 3], norm2)]

    popart_result, popart_weights = _run_popart_test_model(
        data, weights, clipGroups)
    torch_result, torch_weights = _run_torch_test_model(
        data, weights, clipGroups)

    assert popart_result.shape == torch_result.shape
    assert np.allclose(popart_result, torch_result)

    def print_tensor(x):
        x = str(x)
        x = x.replace('\n', '')
        print(f'  {x}')

    for key in popart_weights.keys():
        print(f'{key}:')
        print_tensor(initial_weights[key])
        print_tensor(popart_weights[key])
        print_tensor(torch_weights[key])

    for key in popart_weights.keys():
        assert np.allclose(popart_weights[key], torch_weights[key])
