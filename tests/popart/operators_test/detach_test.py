# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
from op_tester import op_tester
import numpy as np
import popart
import torch
import pytest
import torch.nn as nn
import torch.optim as optim

LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-2


# This example:
#     FWD:                  BWD:
#
#     Input                Input'
#     /   \                 ^
#    /     \               /
#  Pow     Pow           Pow'
#   |       |             ^
#   |       |             |
#   |     Detach          |
#    \     /               \-
#     -Add-                 Add'
#       |                    ^
#       |                    |
#    Softmax              Softmax'
#       |                    ^
#       |                    |
#     Output               Output'
#
# The right branch is not used in the calculation of the gradient.
@pytest.mark.parametrize("inplacing", [True, False])
def test_detach_grad(op_tester, inplacing):
    data = np.random.rand(4, 4).astype(np.float32)
    w_data = np.random.rand(4, 4).astype(np.float32)
    two = np.array([2]).astype(np.float32)

    def init_builder(builder):
        i = builder.addInputTensor(data)
        two_i = builder.aiOnnx.constant(two, "two")
        w = builder.addInitializedInputTensor(w_data)

        a1 = builder.aiOnnx.pow([i, two_i])
        b1 = builder.aiOnnx.pow([i, two_i])
        detach = builder.aiGraphcore.detach([b1])
        o = builder.aiOnnx.sum([a1, detach, w])
        o = builder.aiOnnx.softmax([o], axis=1)

        builder.addOutputTensor(o)

        return [
            o,
            popart.reservedGradientPrefix() + i,
            popart.reservedGradientPrefix() + w,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        x = torch.tensor(data, requires_grad=True)
        w_t = torch.tensor(w_data, requires_grad=True)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.weight_param = w_t

            def forward(self, x):
                pow1 = x**2
                with torch.no_grad():
                    pow2 = x**2
                out = pow1 + pow2 + self.weight_param
                out = torch.softmax(out, dim=1)
                return out

        net = Net()
        out = net(x)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, x.grad, net.weight_param.grad, None]

    op_tester.patterns = ['PowArg0GradOp']
    op_tester.inplacing = inplacing
    op_tester.run(init_builder, reference, 'train')


# Test the 4 possibilities of detaching the right hand branch in popART vs pytorch:
#
#                   Popart
#
#         Match? True    False
#               +-------------+
#          True |Yes      No  |
#  Torch        |             |
#         False | No      Yes |
#               +-------------+


#              Input
#               /  \
#              /    \
#             /      \
#            /        \
#           -          -
#  w1-----Conv1      Conv2-----w2
#           |          |
#           |          |
#        Reshape1   Reshape2
#           |          |
#           |          |
#           |        Detach (?)
#           \          /
#            \        /
#             \      /
#              -    /
#               Add-
#                |
#                |
#             Softmax
#                |
#                |
#               Out
@pytest.mark.parametrize("detach_branch_popart", [True, False])
@pytest.mark.parametrize("detach_branch_pytorch", [True, False])
def test_detach_grad_branches(detach_branch_popart, detach_branch_pytorch):
    # fix the random seed for this test
    np.random.seed(0)
    Batchsize = 8
    Classes = 32

    dshape = [Batchsize, 2, 4, 4]
    lshape = [Batchsize]
    wshape = [2, 2, 3, 3]

    ip_data = np.random.rand(*dshape).astype(np.float32)
    w1_data = np.random.rand(*wshape).astype(np.float32)
    w2_data = np.random.rand(*wshape).astype(np.float32)
    lb_data = np.random.randint(Classes, size=lshape)

    builder = popart.Builder()

    input_ = builder.addInputTensor(popart.TensorInfo("FLOAT", dshape),
                                    "input_i1")

    lb = builder.addInputTensor(popart.TensorInfo("INT32", lshape))
    w1 = builder.addInitializedInputTensor(w1_data)
    w2 = builder.addInitializedInputTensor(w2_data)

    conv1 = builder.aiOnnx.conv([input_, w1],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                strides=[1, 1],
                                debugPrefix="conv")
    r1 = builder.reshape_const(builder.aiOnnx, [conv1], [Batchsize, Classes])
    conv2 = builder.aiOnnx.conv([input_, w2],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                strides=[1, 1],
                                debugPrefix="conv")
    r2 = builder.reshape_const(builder.aiOnnx, [conv2], [Batchsize, Classes])
    if detach_branch_popart:
        r2 = builder.aiGraphcore.detach([r2])

    add = builder.aiOnnx.sum([r1, r2])
    o = builder.aiOnnx.softmax([add], axis=np.size(lb_data.shape))
    loss = builder.aiGraphcore.nllloss([o, lb])

    dataFlow = popart.DataFlow(1, [
        o, loss,
        popart.reservedGradientPrefix() + o,
        popart.reservedGradientPrefix() + input_, w1, w2
    ])

    opts = popart.SessionOptions()
    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFlow=dataFlow,
        loss=loss,
        optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
        userOptions=opts,
        deviceInfo=popart.DeviceManager().createIpuModelDevice({}))

    session.prepareDevice()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({input_: ip_data, lb: lb_data}, anchors)
    session.weightsFromHost()

    # Torch

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(2, 2, 3, padding=[1, 1], bias=False)
            self.conv2 = nn.Conv2d(2, 2, 3, padding=[1, 1], bias=False)
            self.conv1.weight.data = torch.tensor(w1_data)
            self.conv2.weight.data = torch.tensor(w2_data)
            # PyTorch nll loss expects logsoftmax input
            self.sm = nn.LogSoftmax(dim=np.size(lb_data.shape))
            self.nll = nn.NLLLoss()

        def forward(self, x, y):
            x1 = self.conv1(x)
            x1 = torch.reshape(x1, [Batchsize, Classes])
            if detach_branch_pytorch:
                with torch.no_grad():
                    x2 = self.conv2(x)
            else:
                x2 = self.conv2(x)

            x2 = torch.reshape(x2, [Batchsize, Classes])
            x = x1 + x2
            x = self.sm(x)
            x = self.nll(x, y)
            return x

    net = Net()
    optimizer = optim.SGD(net.parameters(),
                          lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY)

    input_t = torch.tensor(ip_data, requires_grad=True, dtype=torch.float32)
    label_t = torch.tensor(lb_data, requires_grad=False, dtype=torch.long)

    for step in range(4):
        print(f"Step {step +1}")
        session.run(stepio)

        # Torch
        #
        optimizer.zero_grad()
        loss = net(input_t, label_t)
        loss.backward()
        optimizer.step()

        print(detach_branch_popart, detach_branch_pytorch)
        print("Popart: w1", np.mean(anchors[w1]))
        print("PyTorch: w1", np.mean(net.conv1.weight.data.numpy()))
        print("Popart: w2", np.mean(anchors[w2]))
        print("PyTorch: w2", np.mean(net.conv2.weight.data.numpy()))

        # Check the weights match if the branches are the same, if not,
        # make sure the right hand branch doesn't match
        if detach_branch_popart == detach_branch_pytorch:
            assert np.allclose(anchors[w1], net.conv1.weight.data.numpy(),
                               1e-4)
            assert np.allclose(anchors[w2], net.conv2.weight.data.numpy(),
                               1e-4)
        else:
            assert not np.allclose(anchors[w2], net.conv2.weight.data.numpy(),
                                   1e-4)


# Single branch, with a detach operations, similar to above:
#  Input
#    |
#    v
#  Conv
#    |
#    v
#  Reshape
#    |
#    v
#  Detach
#    |
#    v
#  Output
#
# We add the gradient of the input to the anchors,
# however with the detach operation, this should not be possible.
def test_detach_error():
    np.random.seed(0)
    Batchsize = 8
    Classes = 32

    dshape = [Batchsize, 2, 4, 4]
    lshape = [Batchsize]
    wshape = [2, 2, 3, 3]

    ip_data = np.random.rand(*dshape).astype(np.float32)
    w1_data = np.random.rand(*wshape).astype(np.float32)
    lb_data = np.random.randint(Classes, size=lshape)

    builder = popart.Builder()

    input_ = builder.addInputTensor(popart.TensorInfo("FLOAT", dshape),
                                    "input_i1")

    lb = builder.addInputTensor(popart.TensorInfo("INT32", lshape))
    w1 = builder.addInitializedInputTensor(w1_data)

    conv1 = builder.aiOnnx.conv([input_, w1],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                strides=[1, 1],
                                debugPrefix="conv")
    o = builder.reshape_const(builder.aiOnnx, [conv1], [Batchsize, Classes])
    o = builder.aiGraphcore.detach([o])

    o = builder.aiOnnx.softmax([o], axis=np.size(lshape))

    loss = builder.aiGraphcore.nllloss([o, lb])

    dataFlow = popart.DataFlow(
        1, [o, loss, popart.reservedGradientPrefix() + input_])
    opts = popart.SessionOptions()
    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=dataFlow,
            loss=loss,
            optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
            userOptions=opts,
            deviceInfo=popart.DeviceManager().createIpuModelDevice({}))

    assert (e_info.value.args[0].startswith(
        f"Anchor tensor `{popart.reservedGradientPrefix() + input_}' not in Ir Tensors."
    ))
