# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
import json
from rnn_helper import RNN_Helper

from pathlib import Path

# `import test_util` requires adding to sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def np_rand(*shape):
    return np.random.rand(*shape).astype(np.float32)


def np_zeros(*shape):
    return np.zeros(shape, dtype=np.float32)


# Compare with onnx implementation with minimal inputs when doing inference
@tu.requires_ipu_model
def test_rnn_onnx(op_tester):
    d1 = np.random.randint(0, 20, size=(2, 2, 3)).astype(np.float32)

    input_size = d1.shape[2]  # (2,2,3)
    hidden_size = 7

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3], 2)
        builder.addOutputTensor(Y_h)
        builder.addOutputTensor(Y)
        return [Y, Y_h]

    def reference(ref_data):
        rnn = RNN_Helper(X=d1, W=d2, R=d3)
        Y, Y_h = rnn.step()

        return [Y.astype(np.float32), Y_h.astype(np.float32)]

    op_tester.run(init_builder, reference, 'infer')


# Compare with Pytorch implementation with minimal inputs when doing inference
@tu.requires_ipu_model
def test_rnn_torch(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    d3 = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3], 2)

        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(ref_data):
        rnn = torch.nn.RNN(input_size, hidden_size, 1)
        rnn.weight_ih_l0.data = torch.tensor(d2[0])
        rnn.weight_hh_l0.data = torch.tensor(d3[0])
        rnn.bias_ih_l0.data.fill_(0)
        rnn.bias_hh_l0.data.fill_(0)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = rnn(a)
        Y = torch.unsqueeze(Y, 1)

        return [Y, Y_h]

    op_tester.run(init_builder, reference, 'infer')


# Compare with Pytorch implementation with minimal inputs when training
@tu.requires_ipu_model
def test_rnn_torch_grad(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    d3 = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3], 2)
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
        rnn = torch.nn.RNN(input_size, hidden_size, 1)
        rnn.weight_ih_l0.data = torch.tensor(d2[0])
        rnn.weight_hh_l0.data = torch.tensor(d3[0])
        rnn.bias_ih_l0.data.fill_(0)
        rnn.bias_hh_l0.data.fill_(0)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = rnn(a)
        Ys = Y.squeeze()
        Y1 = Ys + Y_h

        Y.retain_grad()
        Y_h.retain_grad()
        Ys.retain_grad()
        Y1.retain_grad()

        d__o = ref_data.getOutputTensorGrad(0)
        Y1.backward(torch.tensor(d__o))

        return [
            Y1,
            Y1.grad,
            a.grad,
            rnn.weight_ih_l0.grad.unsqueeze(0),
            rnn.weight_hh_l0.grad.unsqueeze(0),
        ]

    op_tester.run(init_builder, reference, 'train')


# Compare with onnx implementation when using biases in onnx
@tu.requires_ipu_model
def test_rnn_biases_onnx(op_tester):
    d1 = np.random.randint(0, 20, size=(2, 2, 3)).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 5
    num_directions = 1

    d2 = np.random.rand(num_directions, hidden_size,
                        input_size).astype(np.float32)
    d3 = np.random.rand(num_directions, hidden_size,
                        hidden_size).astype(np.float32)
    d4 = np.random.rand(num_directions, 2 * hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3, i4], 2)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(ref_data):
        rnn = RNN_Helper(X=d1, W=d2, R=d3, B=d4)
        Y, Y_h = rnn.step()

        return [Y.astype(np.float32), Y_h.astype(np.float32)]

    op_tester.run(init_builder, reference, 'infer')


# Compare with Pytorch implementation when using biases in inference
@tu.requires_ipu_model
def test_rnn_biases_torch(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    d3 = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    d4 = np.random.rand(1, hidden_size).astype(np.float32)

    d5 = np.random.rand(1, hidden_size).astype(np.float32)

    d6 = np.concatenate((d4, d5), axis=1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d6)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3, i4], 2)

        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(ref_data):
        rnn = torch.nn.RNN(input_size, hidden_size, 1)
        rnn.weight_ih_l0.data = torch.tensor(d2[0])
        rnn.weight_hh_l0.data = torch.tensor(d3[0])
        rnn.bias_ih_l0.data = torch.tensor(d4)
        rnn.bias_hh_l0.data = torch.tensor(d5)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = rnn(a)
        Y = torch.unsqueeze(Y, 1)

        return [Y, Y_h]

    op_tester.run(init_builder, reference, 'infer')


# Commpare with onnx implementation when using initial_h in inference
@tu.requires_ipu_model
def test_rnn_initial_h_onnx(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 5
    num_directions = 1

    d2 = np.random.rand(num_directions, hidden_size,
                        input_size).astype(np.float32)
    d3 = np.random.rand(num_directions, hidden_size,
                        hidden_size).astype(np.float32)
    d4 = np.random.rand(num_directions, hidden_size * 2).astype(np.float32)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np.random.rand(num_directions, batch_size,
                               hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(seq_lens)
        i6 = builder.addInputTensor(initial_h)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3, i4, i5, i6], 2)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(ref_data):
        rnn = RNN_Helper(X=d1, W=d2, R=d3, B=d4, initial_h=initial_h)
        Y, Y_h = rnn.step()

        return [Y.astype(np.float32), Y_h.astype(np.float32)]

    op_tester.run(init_builder, reference, 'infer')


# Test against pytorch implementation with the 2 activation functions supported by Pytorch
@tu.requires_ipu_model
@pytest.mark.parametrize("activation", ["Tanh", "Relu"])
def test_rnn_nondefault_activation(op_tester, activation):
    d1 = np.random.rand(2, 2, 3).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    d3 = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3], 2)
        Ys = builder.aiOnnx.squeeze([Y], [])
        Y1 = builder.aiOnnx.add([Ys, Y_h])
        builder.addNodeAttribute("activations", [activation], {Y, Y_h})
        builder.addOutputTensor(Y1)
        return [
            Y1,
            popart.reservedGradientPrefix() + Y1,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
        ]

    def reference(ref_data):
        rnn = torch.nn.RNN(input_size,
                           hidden_size,
                           1,
                           nonlinearity=activation.lower())
        rnn.weight_ih_l0.data = torch.tensor(d2[0])
        rnn.weight_hh_l0.data = torch.tensor(d3[0])
        rnn.bias_ih_l0.data.fill_(0)
        rnn.bias_hh_l0.data.fill_(0)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = rnn(a)
        Ys = Y.squeeze()
        Y1 = Ys + Y_h

        Y.retain_grad()
        Y_h.retain_grad()
        Ys.retain_grad()
        Y1.retain_grad()

        d__o = ref_data.getOutputTensorGrad(0)
        Y1.backward(torch.tensor(d__o))

        return [
            Y1,
            Y1.grad,
            a.grad,
            rnn.weight_ih_l0.grad.unsqueeze(0),
            rnn.weight_hh_l0.grad.unsqueeze(0),
        ]

    op_tester.run(init_builder, reference, 'train')


# Grad with bias and initial_h inputs
# Compare to pytorch implementation
@tu.requires_ipu_model
def test_rnn_torch_grad_all_inputs(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 2
    num_directions = 1

    input_weights = np.random.rand(1, hidden_size,
                                   input_size).astype(np.float32)

    hidden_weights = np.random.rand(1, hidden_size,
                                    hidden_size).astype(np.float32)

    bi = np.random.rand(1, hidden_size).astype(np.float32)

    bh = np.random.rand(1, hidden_size).astype(np.float32)

    biases = np.concatenate((bi, bh), axis=1)
    input_biases_torch = bi
    hidden_biases_torch = bh

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np.random.rand(num_directions, batch_size,
                               hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(input_weights)
        i3 = builder.addInputTensor(hidden_weights)
        i4 = builder.addInputTensor(biases)
        i5 = builder.addInputTensor(seq_lens)
        i6 = builder.addInputTensor(initial_h)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3, i4, i5, i6], 2)
        Ys = builder.aiOnnx.squeeze([Y], [])
        Y1 = builder.aiOnnx.add([Ys, Y_h])

        builder.addOutputTensor(Y1)
        print([
            Y1,
            popart.reservedGradientPrefix() + Y1,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + i4,
            popart.reservedGradientPrefix() + i6
        ])
        return [
            Y1,
            popart.reservedGradientPrefix() + Y1,
            popart.reservedGradientPrefix() + i1,  # input
            popart.reservedGradientPrefix() + i2,  # input/1
            popart.reservedGradientPrefix() + i3,  # input/2
            popart.reservedGradientPrefix() + i4,  # input/3
            popart.reservedGradientPrefix() + i6  # input/4
        ]

    def reference(ref_data):
        rnn = torch.nn.RNN(input_size, hidden_size, 1)
        rnn.weight_ih_l0.data = torch.tensor(input_weights[0])
        rnn.weight_hh_l0.data = torch.tensor(hidden_weights[0])
        rnn.bias_ih_l0.data = torch.tensor(input_biases_torch)
        rnn.bias_hh_l0.data = torch.tensor(hidden_biases_torch)

        h0 = torch.tensor(initial_h, requires_grad=True)

        a = torch.tensor(d1, requires_grad=True)
        Y, Y_h = rnn(a, h0)
        Ys = Y.squeeze()
        Y1 = Ys + Y_h

        Y.retain_grad()
        Y_h.retain_grad()
        Ys.retain_grad()
        Y1.retain_grad()

        d__o = ref_data.getOutputTensorGrad(0)
        Y1.backward(torch.tensor(d__o))

        wig = rnn.weight_ih_l0.grad
        wig = wig.unsqueeze(0)

        whg = rnn.weight_hh_l0.grad
        whg = whg.unsqueeze(0)

        b_grad = torch.cat((rnn.bias_ih_l0.grad, rnn.bias_hh_l0.grad)).view(
            1, 2 * hidden_size)

        return [Y1, Y1.grad, a.grad, wig, whg, b_grad, h0.grad]

    op_tester.run(init_builder, reference, 'train')


# Check that the following model
#   rnn -> rnn -> rnn -> rnn -> identity
# with identical rnn ops gets outlined so that there is only 1 rnn op
@tu.requires_ipu_model
def test_rnn_outlining(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInitializedInputTensor(d2)
        i3 = builder.addInitializedInputTensor(d3)
        x = i1
        for i in range(4):
            Y, Y_h = builder.aiOnnx.rnn([x, i2, i3], 2)
            x = builder.aiOnnx.squeeze([Y], axes=[1])
        Y = builder.aiOnnx.identity([Y])
        builder.addOutputTensor(Y)
        return [Y]

    def reference(ref_data):
        return [None]

    op_tester.device = tu.create_test_device()
    session = op_tester.run(init_builder, reference, 'train')

    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    main_graph = ir['maingraph']

    # There should be no rnns left in the main graph
    main_graph_rnns = [op for op in main_graph if op['type'] == 'RNN']
    assert len(main_graph_rnns) == 0

    # There should be one rnn left in the whole model
    rnns = []
    for graph in ir.values():
        new_rnn_ops = [op for op in graph if op['type'] == 'RNN']
        rnns.extend(new_rnn_ops)
    assert len(rnns) == 1


# Try an activation not supported by Poplar
@tu.requires_ipu_model
def test_rnn_unsupported_activation(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h = builder.aiOnnx.rnn([i1, i2, i3], 2)
        # Set an invalid activation
        builder.addNodeAttribute("activations", ["Affine"], {Y, Y_h})
        builder.addOutputTensor(Y_h)
        return [Y, Y_h]

    def reference(ref_data):
        # The reference should never run, popart should raise an exception before this.
        assert False

    op_tester.device = tu.create_test_device()
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert 'Affine' in e_info.value.args[0]
    assert 'not supported' in e_info.value.args[0]


# Only one activation is supported, try 0
@tu.requires_ipu_model
def test_rnn_bad_number_of_activations(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y = builder.aiOnnx.rnn([i1, i2, i3], 1)
        # Set invalid number of activations
        builder.addNodeAttribute("activations", [], set(Y))
        builder.addOutputTensor(Y[0])
        return Y

    def reference(ref_data):
        # The reference should never run, popart should raise an exception before this.
        assert False

    op_tester.device = tu.create_test_device()
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert 'only supports 1 activation' in e_info.value.args[0]


# Test that wrong hidden_size attribute throws error
@tu.requires_ipu_model
def test_rnn_wrong_hidden_size(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y = builder.aiOnnx.rnn([i1, i2, i3], 1)
        # Wrong hidden_size attribute
        builder.addNodeAttribute("hidden_size", hidden_size + 1, set(Y))
        return Y

    def reference(ref_data):
        # The reference should never run, popart should raise an exception before this.
        assert False

    op_tester.device = tu.create_test_device()
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert 'hidden_size' in e_info.value.args[0]
    assert 'does not match' in e_info.value.args[0]


# Test that there's no error when hidden_size attribute matches up with the inferred hidden size
@tu.requires_ipu_model
def test_rnn_correct_hidden_size(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y = builder.aiOnnx.rnn([i1, i2, i3], 1)
        # set hidden_size attribute to correct value
        builder.addNodeAttribute("hidden_size", hidden_size, set(Y))
        builder.addOutputTensor(Y[0])
        return Y

    def reference(ref_data):
        # Not checking the output
        return [None]

    op_tester.device = tu.create_test_device()
    op_tester.run(init_builder, reference, 'infer')


# Test activation_alpha attribute throws error
@tu.requires_ipu_model
def test_rnn_activation_alpha_error(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y = builder.aiOnnx.rnn([i1, i2, i3], 1)
        # Set activation_alpha attribute
        builder.addNodeAttribute("activation_alpha", [1.], set(Y))
        return Y

    def reference(ref_data):
        # The reference should never run, popart should raise an exception before this.
        assert False

    op_tester.device = tu.create_test_device()
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert 'activation_alpha' in e_info.value.args[0]
    assert 'not supported' in e_info.value.args[0]


# Test activation_beta attribute throws error
@tu.requires_ipu_model
def test_rnn_activation_beta_error(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y = builder.aiOnnx.rnn([i1, i2, i3], 1)
        # Set activation_beta attribute
        builder.addNodeAttribute("activation_beta", [1.], set(Y))
        return Y

    def reference(ref_data):
        # The reference should never run, popart should raise an exception before this.
        assert False

    op_tester.device = tu.create_test_device()
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert 'activation_beta' in e_info.value.args[0]
    assert 'not supported' in e_info.value.args[0]


# Test clip attribute throws error
@tu.requires_ipu_model
def test_rnn_clip_error(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y = builder.aiOnnx.rnn([i1, i2, i3], 1)
        # Set clip attribute
        builder.addNodeAttribute("clip", 1., set(Y))
        return Y

    def reference(ref_data):
        # The reference should never run, popart should raise an exception before this.
        assert False

    op_tester.device = tu.create_test_device()
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert 'clip' in e_info.value.args[0]
    assert 'not supported' in e_info.value.args[0]


# Test direction attribute throws error when direction != "forward"
@tu.requires_ipu_model
def test_rnn_direction_error(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y = builder.aiOnnx.rnn([i1, i2, i3], 1)
        # Set direction to something other than "forward"
        builder.addNodeAttribute("direction", "bidirectional", set(Y))
        return Y

    def reference(ref_data):
        # The reference should never run, popart should raise an exception before this.
        assert False

    op_tester.device = tu.create_test_device()
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert 'only supports' in e_info.value.args[0]
    assert 'forward' in e_info.value.args[0]


# Test forward direction attribute does not throw error
@tu.requires_ipu_model
def test_rnn_forward_direction(op_tester):
    d1 = np.array([[[1.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 1

    d2 = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y = builder.aiOnnx.rnn([i1, i2, i3], 1)
        # Set direction to "forward"
        builder.addNodeAttribute("direction", "forward", set(Y))
        return Y

    def reference(ref_data):
        return [None]

    op_tester.device = tu.create_test_device()
    op_tester.run(init_builder, reference, 'infer')
