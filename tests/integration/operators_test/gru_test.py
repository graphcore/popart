# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
from rnn_helper import GRU_Helper

from pathlib import Path

# `import test_util` requires adding to sys.path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def np_rand(*shape):
    return np.random.rand(*shape).astype(np.float32)


def np_zeros(*shape):
    return np.zeros(shape, dtype=np.float32)


def test_gru(op_tester):
    d1 = np.random.randint(0, 20, size=(2, 2, 3)).astype(np.float32)

    input_size = d1.shape[2]  # (2,2,3)
    hidden_size = 7

    d2 = np.random.rand(1, 3 * hidden_size, input_size).astype(np.float32)
    d3 = np.random.rand(1, 3 * hidden_size, hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h = builder.aiOnnx.gru([i1, i2, i3], 2, clip=None)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(_):  # ref_data is an unused argument
        gru = GRU_Helper(X=d1, W=d2, R=d3)
        Y, Y_h = gru.step()

        return [Y.astype(np.float32), Y_h.astype(np.float32)]

    op_tester.atol = 1e-06
    op_tester.rtol = 1e-03
    with tu.create_test_device() as device:
        op_tester.device = device
        op_tester.run(init_builder, reference, "infer")


def test_gru_torch(op_tester):
    d1 = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    ).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    wz = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wr = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wh = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    whz = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whr = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whh = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    d2 = np.concatenate((wz, wr, wh), axis=1)
    d2_torch = np.concatenate((wr, wz, wh), axis=1)

    d3 = np.concatenate((whz, whr, whh), axis=1)
    d3_torch = np.concatenate((whr, whz, whh), axis=1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h = builder.aiOnnx.gru([i1, i2, i3], 2, clip=None, linear_before_reset=1)

        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(_):  # ref_data is an unused argument
        gru = torch.nn.GRU(input_size, hidden_size, 1)
        gru.weight_ih_l0.data = torch.tensor(d2_torch[0])
        gru.weight_hh_l0.data = torch.tensor(d3_torch[0])
        gru.bias_ih_l0.data.fill_(0)
        gru.bias_hh_l0.data.fill_(0)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = gru(a)
        Y = torch.unsqueeze(Y, 1)

        return [Y, Y_h]

    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    # No need to relax tolerances!
    op_tester.run(init_builder, reference, "infer")


def test_gru_torch_grad(op_tester):
    d1 = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    ).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    wz = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wr = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wh = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    whz = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whr = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whh = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    d2 = np.concatenate((wz, wr, wh), axis=1)
    d2_torch = np.concatenate((wr, wz, wh), axis=1)

    d3 = np.concatenate((whz, whr, whh), axis=1)
    d3_torch = np.concatenate((whr, whz, whh), axis=1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h = builder.aiOnnx.gru([i1, i2, i3], 2, clip=None)
        Ys = builder.aiOnnx.squeeze([Y], [])
        Y1 = builder.aiOnnx.add([Ys, Y_h])

        builder.addOutputTensor(Y1)
        return [
            Y1,
            popart.reservedGradientPrefix() + Y1,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
        ]

    def reference(ref_data):
        gru = torch.nn.GRU(input_size, hidden_size, 1)
        gru.weight_ih_l0.data = torch.tensor(d2_torch[0])
        gru.weight_hh_l0.data = torch.tensor(d3_torch[0])
        gru.bias_ih_l0.data.fill_(0)
        gru.bias_hh_l0.data.fill_(0)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = gru(a)
        Ys = Y.squeeze()
        Y1 = Ys + Y_h

        Y.retain_grad()
        Y_h.retain_grad()
        Ys.retain_grad()
        Y1.retain_grad()

        d__o = ref_data.getOutputTensorGrad(0)
        Y1.backward(torch.tensor(d__o))

        # reorder the weights for comparison with popart
        wr, wz, wh = torch.split(gru.weight_ih_l0.grad, hidden_size)
        wig = torch.cat((wz, wr, wh), dim=0)
        wig.unsqueeze_(0)

        # reorder the weights for comparison with popart
        wr, wz, wh = torch.split(gru.weight_hh_l0.grad, hidden_size)
        whg = torch.cat((wz, wr, wh), dim=0)
        whg.unsqueeze_(0)

        return [Y1, Y1.grad, a.grad, wig, whg, None]

    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)

    op_tester.atol = 1e-06
    op_tester.rtol = 1e-05
    op_tester.run(init_builder, reference, "train")


def test_gru_biases(op_tester):
    d1 = np.random.randint(0, 20, size=(2, 2, 3)).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 5
    num_directions = 1

    d2 = np.random.rand(num_directions, 3 * hidden_size, input_size).astype(np.float32)
    d3 = np.random.rand(num_directions, 3 * hidden_size, hidden_size).astype(np.float32)
    d4 = np.random.rand(num_directions, 6 * hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        Y, Y_h = builder.aiOnnx.gru([i1, i2, i3, i4], 2, clip=None)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(_):  # ref_data is an unused argument
        lstm = GRU_Helper(X=d1, W=d2, R=d3, B=d4)
        Y, Y_h = lstm.step()

        return [Y.astype(np.float32), Y_h.astype(np.float32)]

    op_tester.atol = 1e-06
    op_tester.rtol = 1e-03
    op_tester.run(init_builder, reference, "infer")


def test_gru_biases_torch(op_tester):
    d1 = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    ).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    wz = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wr = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wh = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    whz = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whr = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whh = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    bz = np.random.rand(1, hidden_size).astype(np.float32)
    br = np.random.rand(1, hidden_size).astype(np.float32)
    bh = np.random.rand(1, hidden_size).astype(np.float32)

    bhz = np.random.rand(1, hidden_size).astype(np.float32)
    bhr = np.random.rand(1, hidden_size).astype(np.float32)
    bhh = np.random.rand(1, hidden_size).astype(np.float32)

    d2 = np.concatenate((wz, wr, wh), axis=1)
    d2_torch = np.concatenate((wr, wz, wh), axis=1)

    d3 = np.concatenate((whz, whr, whh), axis=1)
    d3_torch = np.concatenate((whr, whz, whh), axis=1)

    d4 = np.concatenate((bz, br, bh), axis=1)
    d4_torch = np.concatenate((br, bz, bh), axis=1)

    d5 = np.concatenate((bhz, bhr, bhh), axis=1)
    d5_torch = np.concatenate((bhr, bhz, bhh), axis=1)

    d6 = np.concatenate((d4, d5), axis=1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d6)
        Y, Y_h = builder.aiOnnx.gru(
            [i1, i2, i3, i4], 2, clip=None, linear_before_reset=1
        )

        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(_):  # ref_data is an unused argument
        gru = torch.nn.GRU(input_size, hidden_size, 1)
        gru.weight_ih_l0.data = torch.tensor(d2_torch[0])
        gru.weight_hh_l0.data = torch.tensor(d3_torch[0])
        gru.bias_ih_l0.data = torch.tensor(d4_torch)
        gru.bias_hh_l0.data = torch.tensor(d5_torch)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = gru(a)
        Y = torch.unsqueeze(Y, 1)

        return [Y, Y_h]

    op_tester.run(init_builder, reference, "infer")


def test_gru_initial_hc(op_tester):
    d1 = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    ).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 5
    num_directions = 1

    d2 = np.random.rand(num_directions, 3 * hidden_size, input_size).astype(np.float32)
    d3 = np.random.rand(num_directions, 3 * hidden_size, hidden_size).astype(np.float32)
    d4 = np.random.rand(num_directions, 6 * hidden_size).astype(np.float32)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np.random.rand(num_directions, batch_size, hidden_size).astype(
        np.float32
    )

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(seq_lens)
        i6 = builder.addInputTensor(initial_h)
        Y, Y_h = builder.aiOnnx.gru([i1, i2, i3, i4, i5, i6], 2)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(_):  # ref_data is an unused argument
        lstm = GRU_Helper(X=d1, W=d2, R=d3, B=d4, initial_h=initial_h)
        Y, Y_h = lstm.step()

        return [Y, Y_h]

    op_tester.run(init_builder, reference, "infer")


def test_gru_torch_grad_all_inputs(op_tester):
    d1 = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    ).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 2
    num_directions = 1

    wz = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wr = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wh = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    whz = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whr = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whh = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    input_weights = np.concatenate((wz, wr, wh), axis=1)
    input_weights_torch = np.concatenate((wr, wz, wh), axis=1)

    hidden_weights = np.concatenate((whz, whr, whh), axis=1)
    hidden_weights_torch = np.concatenate((whr, whz, whh), axis=1)

    biz = np.random.rand(1, hidden_size).astype(np.float32)
    bir = np.random.rand(1, hidden_size).astype(np.float32)
    bih = np.random.rand(1, hidden_size).astype(np.float32)

    bhz = np.random.rand(1, hidden_size).astype(np.float32)
    bhr = np.random.rand(1, hidden_size).astype(np.float32)
    bhh = np.random.rand(1, hidden_size).astype(np.float32)

    biases = np.concatenate((biz, bir, bih, bhz, bhr, bhh), axis=1)
    input_biases_torch = np.concatenate((bir, biz, bih), axis=1)
    hidden_biases_torch = np.concatenate((bhr, bhz, bhh), axis=1)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np.random.rand(num_directions, batch_size, hidden_size).astype(
        np.float32
    )

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(input_weights)
        i3 = builder.addInputTensor(hidden_weights)
        i4 = builder.addInputTensor(biases)
        i5 = builder.addInputTensor(seq_lens)
        i6 = builder.addInputTensor(initial_h)
        Y, Y_h = builder.aiOnnx.gru([i1, i2, i3, i4, i5, i6], 2, linear_before_reset=1)
        Ys = builder.aiOnnx.squeeze([Y], [])
        Y1 = builder.aiOnnx.add([Ys, Y_h])

        builder.addOutputTensor(Y1)
        print(
            [
                Y1,
                popart.reservedGradientPrefix() + Y1,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + i2,
                popart.reservedGradientPrefix() + i3,
                popart.reservedGradientPrefix() + i4,
                popart.reservedGradientPrefix() + i6,
            ]
        )
        return [
            Y1,
            popart.reservedGradientPrefix() + Y1,
            popart.reservedGradientPrefix() + i1,  # input
            popart.reservedGradientPrefix() + i2,  # input/1
            popart.reservedGradientPrefix() + i3,  # input/2
            popart.reservedGradientPrefix() + i4,  # input/3
            popart.reservedGradientPrefix() + i6,  # input/4
        ]

    def reference(ref_data):
        gru = torch.nn.GRU(input_size, hidden_size, 1)
        gru.weight_ih_l0.data = torch.tensor(input_weights_torch[0])
        gru.weight_hh_l0.data = torch.tensor(hidden_weights_torch[0])
        gru.bias_ih_l0.data = torch.tensor(input_biases_torch)
        gru.bias_hh_l0.data = torch.tensor(hidden_biases_torch)

        h0 = torch.tensor(initial_h, requires_grad=True)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = gru(a, h0)
        Ys = Y.squeeze()
        Y1 = Ys + Y_h

        Y.retain_grad()
        Y_h.retain_grad()
        Ys.retain_grad()
        Y1.retain_grad()

        d__o = ref_data.getOutputTensorGrad(0)
        Y1.backward(torch.tensor(d__o))

        # reorder the weights for comparison with popart
        wr, wz, wh = torch.split(gru.weight_ih_l0.grad, hidden_size)
        wig = torch.cat((wz, wr, wh), dim=0)
        wig.unsqueeze_(0)

        # reorder the weights for comparison with popart
        wr, wz, wh = torch.split(gru.weight_hh_l0.grad, hidden_size)
        whg = torch.cat((wz, wr, wh), dim=0)
        whg.unsqueeze_(0)

        # reorder the biases for comparison with popart
        bir, biz, bih = torch.split(gru.bias_ih_l0.grad, hidden_size, dim=1)
        bhr, bhz, bhh = torch.split(gru.bias_hh_l0.grad, hidden_size, dim=1)
        b_grad = torch.cat((biz, bir, bih, bhz, bhr, bhh)).view(1, 6 * hidden_size)

        return [Y1, Y1.grad, a.grad, wig, whg, b_grad, h0.grad]

    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.atol = 1e-05
    op_tester.rtol = 1e-05
    op_tester.run(init_builder, reference, "train")


if __name__ == "__main__":
    builder = popart.Builder()
    d1 = np.random.randint(0, 20, size=(2, 2, 3)).astype(np.float32)

    input_size = d1.shape[2]  # (2,2,3)
    hidden_size = 7

    d2 = np.random.rand(1, 3 * hidden_size, input_size).astype(np.float32)
    d3 = np.random.rand(1, 3 * hidden_size, hidden_size).astype(np.float32)

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", d1.shape))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", d2.shape))
    i3 = builder.addInputTensor(popart.TensorInfo("FLOAT", d3.shape))
    Y, Y_h = builder.aiOnnx.gru([i1, i2, i3], 2, clip=None, direction="bidirectional")
    builder.addOutputTensor(Y)

    dataFlow = popart.DataFlow(1, {Y: popart.AnchorReturnType("All")})

    # Create a session to compile and the graph for inference
    # ------------------------------------------------------------------------------
    inferenceOptions = popart.SessionOptions()
    # Need to compile the inference graph with variable weights we they can be updated
    # before execution

    inferenceSession = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=dataFlow,
        userOptions=inferenceOptions,
        deviceInfo=popart.DeviceManager().createIpuModelDevice({}),
    )

    # Compile graph
    inferenceSession.prepareDevice()

    # Create buffers to receive results from the execution
    inferenceAnchors = inferenceSession.initAnchorArrays()
