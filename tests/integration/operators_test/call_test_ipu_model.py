# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import re

import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# Test that errors are thrown when a user created subgraph has ops
# without virtual graph ids, or have incorrect virtual graph ids.
@tu.requires_ipu_model
def test_call_pipelined_error():
    def run_test(force_error):
        builder = popart.Builder()

        i0 = builder.addInputTensor(popart.TensorInfo("INT32", [2]))
        i1 = builder.addInputTensor(popart.TensorInfo("INT32", [2]))

        subgraph_builder = builder.createSubgraphBuilder()

        info = popart.TensorInfo("INT32", [2])
        sgi0 = subgraph_builder.addInputTensor(info)
        sgi1 = subgraph_builder.addInputTensor(info)

        if force_error == 'no_vgraph_id':
            subgraph_builder.addOutputTensor(
                subgraph_builder.aiOnnx.add([sgi0, sgi1]))
        elif force_error == 'wrong_vgraph_id':
            with subgraph_builder.virtualGraph(0):
                subgraph_builder.addOutputTensor(
                    subgraph_builder.aiOnnx.add([sgi0, sgi1]))
        else:
            raise Exception(f'Unknown value for `force_error`: {force_error}')

        with builder.virtualGraph(0), builder.pipelineStage(0):
            act = builder.aiGraphcore.call([i0, i1], 1, subgraph_builder)[0]
        with builder.virtualGraph(1), builder.pipelineStage(1):
            out = builder.aiGraphcore.call([act, i1], 1, subgraph_builder)[0]

        builder.addOutputTensor(out)

        opts = popart.SessionOptions()
        opts.enablePipelining = True
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        with tu.create_test_device(numIpus=4, tilesPerIPU=20) as device:
            with pytest.raises(popart.popart_exception) as e_info:
                _ = popart.InferenceSession(fnModel=builder.getModelProto(),
                                            dataFlow=popart.DataFlow(
                                                10, [out]),
                                            userOptions=opts,
                                            deviceInfo=device)

        print(e_info.value.args[0])

        error_pattern = {
            'no_vgraph_id':
            ('Op Op(.*) in subgraph ".*" does not have a virtual graph id. '
             'When pipelining, subgraph ops must have a virtual graph id set.'
             ),
            'wrong_vgraph_id':
            ('The virtual graph id .* for Op Op.* in subgraph .* does not match '
             'the virtual graph id .* of the calling op. When pipelining, subgraph '
             'ops must have a virtual graph id matching the calling op.')
        }[force_error]
        assert re.match(error_pattern, e_info.value.args[0])

    run_test('no_vgraph_id')
    run_test('wrong_vgraph_id')


# Check that call ops created using the builder can be used when pipelining.
@tu.requires_ipu_model
def test_call_pipelined():
    builder = popart.Builder()

    i0 = builder.addInputTensor(popart.TensorInfo("INT32", [2]))
    i1 = builder.addInputTensor(popart.TensorInfo("INT32", [2]))

    def create_subgraph(vgraph):
        subgraph_builder = builder.createSubgraphBuilder()

        info = popart.TensorInfo("INT32", [2])
        sgi0 = subgraph_builder.addInputTensor(info)
        sgi1 = subgraph_builder.addInputTensor(info)

        with subgraph_builder.virtualGraph(vgraph):
            subgraph_builder.addOutputTensor(
                subgraph_builder.aiOnnx.add([sgi0, sgi1]))
        return subgraph_builder

    with builder.virtualGraph(0), builder.pipelineStage(0):
        act = builder.aiGraphcore.call([i0, i1], 1, create_subgraph(0))[0]
    with builder.virtualGraph(1), builder.pipelineStage(1):
        out = builder.aiGraphcore.call([act, i1], 1, create_subgraph(1))[0]

    builder.addOutputTensor(out)

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=4, tilesPerIPU=20) as device:
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFlow=popart.DataFlow(10, [out]),
                                          userOptions=opts,
                                          deviceInfo=device)

        session.prepareDevice()


# Check that call ops created using the builder can be used when pipelining.
@tu.requires_ipu_model
def test_subgraph_attrs():
    builder = popart.Builder()
    np.random.seed(0)
    numLayers = 3
    batches_per_step = 10
    learning_rate = 0.1
    shape = [4, 4]

    input_ = np.random.rand(*shape).astype('float32')
    w0_init = np.random.rand(*shape).astype('float32')
    lbl = np.random.randint(0, 4, size=[4]).astype(np.int32)

    in0 = builder.addInputTensor(popart.TensorInfo(input_))
    lbl0 = builder.addInputTensor(popart.TensorInfo(lbl))

    def create_subgraph(vgraph):
        subgraph_builder = builder.createSubgraphBuilder()
        info = popart.TensorInfo("FLOAT", shape)
        sgi0 = subgraph_builder.addInputTensor(info)
        sgi1 = subgraph_builder.addInputTensor(info)
        with subgraph_builder.virtualGraph(vgraph):
            matmul = subgraph_builder.aiOnnx.matmul([sgi0, sgi1], "mm_layer")
            relu = subgraph_builder.aiOnnx.relu([matmul], "relu_layer")
            sm = subgraph_builder.aiOnnx.logsoftmax([relu], axis=1)
        subgraph_builder.addOutputTensor(sm)
        return subgraph_builder

    actIn = in0
    for layer in range(numLayers):
        print(f"Layer: {layer}")
        with builder.virtualGraph(layer):
            w0 = builder.addInitializedInputTensor(w0_init)
            actIn = builder.aiGraphcore.call([actIn, w0], 1,
                                             create_subgraph(layer),
                                             f"subgraph_{layer}")[0]
            print(f"\t{actIn}, layer {layer} created on vgraph {layer}")
            assert builder.getVirtualGraph(actIn) == layer

    builder.addOutputTensor(actIn)
    opts = popart.SessionOptions()

    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.enablePipelining = True

    dataFlow = popart.DataFlow(batches_per_step, {
        actIn: popart.AnchorReturnType("All"),
        w0: popart.AnchorReturnType("All")
    })

    lossId = builder.aiGraphcore.nllloss([actIn, lbl0])
    builder.virtualGraph(lossId, 2)

    builder.addOutputTensor(lossId)

    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFlow=dataFlow,
        loss=lossId,
        optimizer=popart.ConstSGD(learning_rate),
        deviceInfo=popart.DeviceManager().createIpuModelDevice(
            {'numIPUs': numLayers}),
        userOptions=opts)

    session.prepareDevice()
    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO(
        {
            in0: np.repeat(input_[np.newaxis, :, :], batches_per_step, axis=0),
            lbl0: np.repeat(lbl[np.newaxis, :], batches_per_step, axis=0)
        }, anchors)

    class Net(nn.Module):
        def __init__(self, layers):
            super(Net, self).__init__()
            self.layers = layers
            self.w_t = nn.Parameter(torch.tensor(w0_init))
            self.nll = nn.NLLLoss()
            self.sm = nn.LogSoftmax(dim=1)

        def forward(self, x, y):
            r = x
            for _ in range(self.layers):
                r = torch.matmul(r, self.w_t)
                r = torch.relu(r)
                r = self.sm(r)

            r = self.nll(r, y)
            return r

    net = Net(numLayers)

    optimizer = optim.SGD(
        net.parameters(),
        lr=learning_rate,
    )

    input_t = torch.tensor(input_, requires_grad=True, dtype=torch.float32)
    label_t = torch.tensor(lbl, requires_grad=False, dtype=torch.long)

    for step in range(4):
        print(f"Step {step +1}")
        session.weightsFromHost()
        session.run(stepio)
        for _ in range(batches_per_step):
            # Torch
            optimizer.zero_grad()
            loss = net(input_t, label_t)
            loss.backward()
            optimizer.step()

        print(loss.detach().numpy(), -np.mean(anchors[actIn]))
        assert np.allclose(loss.detach().numpy(), -1 * np.mean(anchors[actIn]))
        assert np.allclose(net.w_t.data.numpy(), anchors[w0])
