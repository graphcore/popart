# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import onnx
import json
from op_tester import op_tester

from pathlib import Path
from test_session import PopartTestSession

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def np_rand(*shape):
    return np.random.rand(*shape).astype(np.float32)


def np_zeros(*shape):
    return np.zeros(shape, dtype=np.float32)


def test_lstm(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 7

    d2 = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, 4 * hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3], 3, clip=None)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        lstm = LSTM_Helper(X=d1, W=d2, R=d3)
        Y, Y_h, Y_c = lstm.step()

        return [Y, Y_h, Y_c]

    op_tester.device = tu.create_test_device()
    op_tester.run(init_builder, reference, 'infer')


# Check the conversion from onnx lstm to popart lstm works.
def test_lstm_popart(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 7

    d2 = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, 4 * hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInitializedInputTensor(d2)
        i3 = builder.addInitializedInputTensor(d3)
        Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3], 3, clip=None)
        builder.addOutputTensor(Y)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        lstm = LSTM_Helper(X=d1, W=d2, R=d3)
        Y, Y_h, Y_c = lstm.step()

        return [Y, Y_h, Y_c]

    op_tester.device = tu.create_test_device()
    op_tester.passes = ['LSTMOp', 'SplitGradOpToConcat']
    session = op_tester.run(init_builder, reference, 'train')

    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    graph = ir['maingraph']

    # There should be one lstm and it should be the aigraphcore lstm
    lstms = [op for op in graph if op['type'] == 'LSTM']
    assert len(lstms) == 1
    assert lstms[0]['domain'] == 'ai.graphcore'


def test_lstm_outlining(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    d2 = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, 4 * hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInitializedInputTensor(d2)
        i3 = builder.addInitializedInputTensor(d3)
        x = i1
        for i in range(4):
            Y, Y_h, Y_c = builder.aiOnnx.lstm([x, i2, i3], 3, clip=None)
            x = builder.aiOnnx.squeeze([Y])
        Y = builder.aiOnnx.identity([Y])
        builder.addOutputTensor(Y)
        return [Y]

    def reference(ref_data):
        return [None]

    op_tester.device = tu.create_test_device()
    op_tester.passes = ['LSTMOp', 'SplitGradOpToConcat']
    session = op_tester.run(init_builder, reference, 'train')

    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    main_graph = ir['maingraph']

    # There should be no lstms left in the main graph
    main_graph_lstms = [op for op in main_graph if op['type'] == 'LSTM']
    assert len(main_graph_lstms) == 0

    # There should be one lstm left in the whole model
    lstms = []
    for graph in ir.values():
        x = [op for op in graph if op['type'] == 'LSTM']
        lstms.extend(x)
    assert len(lstms) == 1


# Check the output of the onnx lstm vs the popart lstm.
# Weights are transformed outside of popart.
def test_lstm_onnx_vs_popart():
    def run_lstm(lstm_type, inputs):
        def init_builder(builder):
            input_ids = [builder.addInputTensor(i) for i in inputs]
            if lstm_type == 'onnx':
                output_ids = builder.aiOnnx.lstm(input_ids, 3, clip=None)
            elif lstm_type == 'popart':
                output_ids = builder.aiGraphcore.lstm(input_ids)
            else:
                raise SystemError(f"unhandled lstm type '{lstm_type}'")
            return [output_ids[0]]

        session = PopartTestSession()
        anchors = session.prepare_and_run(init_builder)

        assert len(anchors) == 1
        anchors = [v for v in anchors.values()]
        return anchors[0]

    def reshape_weight_for_popart(onnx_weights):
        hidden_size = onnx_weights.shape[1] // 4

        def transform_chunk(idx):
            x = onnx_weights[0, idx * hidden_size:(idx + 1) * hidden_size, :]
            x = x.transpose()
            x = np.expand_dims(x, 0)
            return x

        chunks = [transform_chunk(i) for i in range(4)]
        out = np.concatenate((chunks[2], chunks[0], chunks[3], chunks[1]))
        return np.ascontiguousarray(out)

    num_directions = 1
    seq_length = 5
    batch_size = 2
    input_size = 3
    hidden_size = 7

    data = np_rand(seq_length, batch_size, input_size)
    onnx_input_weights = np_rand(1, 4 * hidden_size, input_size)
    onnx_output_weights = np_rand(1, 4 * hidden_size, hidden_size)
    onnx_biases = np_rand(1, 8 * hidden_size)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np_rand(num_directions, batch_size, hidden_size)
    initial_c = np_rand(num_directions, batch_size, hidden_size)

    popart_input_weights = reshape_weight_for_popart(onnx_input_weights)
    popart_output_weights = reshape_weight_for_popart(onnx_output_weights)

    chunks = [
        onnx_biases[:, i * hidden_size:(i + 1) * hidden_size] for i in range(8)
    ]
    popart_biases = np.concatenate([chunks[i] for i in (2, 0, 3, 1)])
    popart_biases += np.concatenate([chunks[i] for i in (6, 4, 7, 5)])

    popart_weights = np.concatenate(
        (popart_input_weights, popart_output_weights), axis=1)
    popart_initial_state = np.concatenate((initial_h, initial_c))

    onnx_out = run_lstm('onnx', [
        data, onnx_input_weights, onnx_output_weights, onnx_biases, seq_lens,
        initial_h, initial_c
    ])
    onnx_out = np.squeeze(onnx_out)
    popart_out = run_lstm(
        'popart', [data, popart_weights, popart_biases, popart_initial_state])

    print('Onnx output:')
    print(onnx_out)
    print()
    print('Popart output:')
    print(popart_out)
    assert np.array_equal(onnx_out, popart_out)


# Check the output of the onnx lstm vs the popart lstm.
# Weights are transformed inside popart.
def test_lstm_onnx_vs_popart_2():
    def run_onnx_lstm(inputs):
        def init_builder(builder):
            input_ids = [builder.addInputTensor(i) for i in inputs]
            Y, Y_h, Y_c = builder.aiOnnx.lstm(input_ids, 3, clip=None)
            return [Y]

        session = PopartTestSession()
        anchors = session.prepare_and_run(init_builder)

        assert len(anchors) == 1
        anchors = [v for v in anchors.values()]
        return anchors[0]

    def run_popart_lstm(data, input_weights, output_weights, biases, initial_h,
                        initial_c):
        def init_builder(builder):
            tData = builder.addInputTensor(data, 'data')
            tIW = builder.addInputTensor(input_weights, 'input_weights')
            tOW = builder.addInputTensor(output_weights, 'output_weights')

            def reshape_weights(w):
                ws = builder.aiOnnx.split([w], 4, 1, [hidden_size] * 4)
                ws = [builder.aiOnnx.transpose([i], [0, 2, 1]) for i in ws]
                ws = builder.aiOnnx.concat([ws[i] for i in (2, 0, 3, 1)], 0)
                return ws

            tIW = reshape_weights(tIW)
            tOW = reshape_weights(tOW)
            tWeights = builder.aiOnnx.concat([tIW, tOW], 1)

            tBiases = builder.addInputTensor(biases, 'biases')
            tBiases = builder.aiOnnx.split([tBiases], 8, 1, [hidden_size] * 8)
            tBiases0 = builder.aiOnnx.concat(
                [tBiases[i] for i in (2, 0, 3, 1)], 0)
            tBiases1 = builder.aiOnnx.concat(
                [tBiases[i] for i in (6, 4, 7, 5)], 0)
            tBiases = builder.aiOnnx.add([tBiases0, tBiases1])

            tInitH = builder.addInputTensor(initial_h, 'initial_h')
            tInitC = builder.addInputTensor(initial_c, 'initial_c')
            tInitState = builder.aiOnnx.concat([tInitH, tInitC], 0)

            input_ids = [tData, tWeights, tBiases, tInitState]
            output_ids = builder.aiGraphcore.lstm(input_ids)
            return [output_ids[0]]

        session = PopartTestSession()
        anchors = session.prepare_and_run(init_builder)

        assert len(anchors) == 1
        anchors = [v for v in anchors.values()]
        return anchors[0]

    def reshape_weight_for_popart(onnx_weights):
        hidden_size = onnx_weights.shape[1] // 4

        def transform_chunk(idx):
            x = onnx_weights[0, idx * hidden_size:(idx + 1) * hidden_size, :]
            x = x.transpose()
            x = np.expand_dims(x, 0)
            return x

        chunks = [transform_chunk(i) for i in range(4)]
        out = np.concatenate((chunks[2], chunks[0], chunks[3], chunks[1]))
        return np.ascontiguousarray(out)

    num_directions = 1
    seq_length = 2
    batch_size = 5
    input_size = 3
    hidden_size = 7

    data = np_rand(seq_length, batch_size, input_size)
    onnx_input_weights = np_rand(1, 4 * hidden_size, input_size)
    onnx_output_weights = np_rand(1, 4 * hidden_size, hidden_size)
    onnx_biases = np_rand(1, 8 * hidden_size)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np_rand(num_directions, batch_size, hidden_size)
    initial_c = np_rand(num_directions, batch_size, hidden_size)

    onnx_out = run_onnx_lstm([
        data, onnx_input_weights, onnx_output_weights, onnx_biases, seq_lens,
        initial_h, initial_c
    ])
    onnx_out = np.squeeze(onnx_out)
    popart_out = run_popart_lstm(data, onnx_input_weights, onnx_output_weights,
                                 onnx_biases, initial_h, initial_c)

    print('Onnx output:')
    print(onnx_out)
    print()
    print('Popart output:')
    print(popart_out)
    assert np.array_equal(onnx_out, popart_out)


# Check the output of the onnx lstm vs the popart lstm during training.
# Weights are transformed inside popart.
def test_lstm_training_onnx_vs_popart():
    data_id = 'data'
    input_weights_id = 'inputWeights'
    output_weights_id = 'outputWeights'
    biases_id = 'biases'
    init_h_id = 'initH'
    init_c_id = 'initC'

    def run_onnx_lstm(data, input_weights, output_weights, biases, seq_lens,
                      initial_h, initial_c, sum_outputs):
        def init_builder(builder):
            tData = builder.addInputTensor(data, data_id)
            tIW = builder.addInitializedInputTensor(input_weights,
                                                    input_weights_id)
            tOW = builder.addInitializedInputTensor(output_weights,
                                                    output_weights_id)
            tBiases = builder.addInitializedInputTensor(biases, biases_id)
            tSeqLens = builder.addInputTensor(seq_lens)
            tInitH = builder.addInputTensor(initial_h, init_h_id)
            tInitC = builder.addInputTensor(initial_c, init_c_id)
            Y, Y_h, Y_c = builder.aiOnnx.lstm(
                [tData, tIW, tOW, tBiases, tSeqLens, tInitH, tInitC],
                3,
                clip=None)
            out = Y
            if sum_outputs:
                out = builder.aiOnnx.add([Y, Y_c], "sum_out_and_cell_state")

            return [
                out,
                popart.reservedGradientPrefix() + tData,
                popart.reservedGradientPrefix() + tIW,
                popart.reservedGradientPrefix() + tOW,
                popart.reservedGradientPrefix() + tBiases
            ]

        session = PopartTestSession()
        session.mode = 'train'
        anchors = session.prepare_and_run(init_builder)

        return anchors

    def run_popart_lstm(data, input_weights, output_weights, biases, initial_h,
                        initial_c, sum_outputs):
        def init_builder(builder):
            tData = builder.addInputTensor(data, data_id)
            tIW = builder.addInitializedInputTensor(input_weights,
                                                    input_weights_id)
            tOW = builder.addInitializedInputTensor(output_weights,
                                                    output_weights_id)
            tBiases = builder.addInitializedInputTensor(biases, biases_id)
            tInitH = builder.addInputTensor(initial_h, init_h_id)
            tInitC = builder.addInputTensor(initial_c, init_c_id)

            def reshape_weights(w):
                ws = builder.aiOnnx.split([w], 4, 1, [hidden_size] * 4)
                ws = [builder.aiOnnx.transpose([i], [0, 2, 1]) for i in ws]
                ws = builder.aiOnnx.concat([ws[i] for i in (2, 0, 3, 1)], 0)
                return ws

            tIW = reshape_weights(tIW)
            tOW = reshape_weights(tOW)
            tWeights = builder.aiOnnx.concat([tIW, tOW], 1)

            tBiases = builder.aiOnnx.split([tBiases], 8, 1, [hidden_size] * 8)
            tBiases0 = builder.aiOnnx.concat(
                [tBiases[i] for i in (2, 0, 3, 1)], 0)
            tBiases1 = builder.aiOnnx.concat(
                [tBiases[i] for i in (6, 4, 7, 5)], 0)
            tBiases = builder.aiOnnx.add([tBiases0, tBiases1])

            tInitState = builder.aiOnnx.concat([tInitH, tInitC], 0)

            input_ids = [tData, tWeights, tBiases, tInitState]
            out, cell_state = builder.aiGraphcore.lstm(input_ids)

            if sum_outputs:
                out = builder.aiOnnx.add([out, cell_state],
                                         "sum_out_and_cell_state")

            return [
                out,
                popart.reservedGradientPrefix() + data_id,
                popart.reservedGradientPrefix() + input_weights_id,
                popart.reservedGradientPrefix() + output_weights_id,
                popart.reservedGradientPrefix() + biases_id,
            ]

        session = PopartTestSession()
        session.mode = 'train'
        anchors = session.prepare_and_run(init_builder)
        return anchors

    num_directions = 1
    seq_length = 2
    batch_size = 2
    input_size = 3
    hidden_size = 7

    data = np_rand(seq_length, batch_size, input_size)
    onnx_input_weights = np_rand(1, 4 * hidden_size, input_size)
    onnx_output_weights = np_rand(1, 4 * hidden_size, hidden_size)
    onnx_biases = np_rand(1, 8 * hidden_size)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np_rand(num_directions, batch_size, hidden_size)
    initial_c = np_rand(num_directions, batch_size, hidden_size)

    # Run with sum_outputs set to True
    for sum_outputs in [True, False]:
        onnx_out = run_onnx_lstm(data, onnx_input_weights, onnx_output_weights,
                                 onnx_biases, seq_lens, initial_h, initial_c,
                                 sum_outputs)
        popart_out = run_popart_lstm(data, onnx_input_weights,
                                     onnx_output_weights, onnx_biases,
                                     initial_h, initial_c, sum_outputs)

        for k, ov in onnx_out.items():
            if k.startswith(popart.reservedGradientPrefix()):
                print(f'Checking anchor {k}')

                if k == 'model_out':
                    ov = np.squeeze(ov)
                pv = popart_out[k]
                assert np.array_equal(ov, pv)


def test_lstm_torch(op_tester):
    input_size = 3
    seq_length = 5
    d1 = np.arange(seq_length * input_size, dtype=np.float32) * 0.05
    d1 = d1.reshape(seq_length, 1, input_size)

    hidden_size = 3

    wi = np_rand(1, hidden_size, input_size)
    wo = np_rand(1, hidden_size, input_size)
    wf = np_rand(1, hidden_size, input_size)
    wc = np_rand(1, hidden_size, input_size)

    whi = np_rand(1, hidden_size, hidden_size)
    who = np_rand(1, hidden_size, hidden_size)
    whf = np_rand(1, hidden_size, hidden_size)
    whc = np_rand(1, hidden_size, hidden_size)

    d2 = np.concatenate((wi, wo, wf, wc), axis=1)
    d2_torch = np.concatenate((wi, wf, wc, wo), axis=1)

    d3 = np.concatenate((whi, who, whf, whc), axis=1)
    d3_torch = np.concatenate((whi, whf, whc, who), axis=1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3], 3)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        lstm = torch.nn.LSTM(input_size, hidden_size, 1)
        lstm.weight_ih_l0.data = torch.tensor(d2_torch[0])
        lstm.weight_hh_l0.data = torch.tensor(d3_torch[0])
        lstm.bias_ih_l0.data.fill_(0)
        lstm.bias_hh_l0.data.fill_(0)

        a = torch.tensor(d1, requires_grad=True)
        Y, (Y_h, Y_c) = lstm(a)
        Y = torch.unsqueeze(Y, 1)

        return [Y, Y_h, Y_c]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')


def test_lstm_torch_grad(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 3

    wi = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wo = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wf = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wc = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    whi = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    who = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whf = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whc = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    d2 = np.concatenate((wi, wo, wf, wc), axis=1)
    d2_torch = np.concatenate((wi, wf, wc, wo), axis=1)

    d3 = np.concatenate((whi, who, whf, whc), axis=1)
    d3_torch = np.concatenate((whi, whf, whc, who), axis=1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3], 3)
        Ys = builder.aiOnnx.squeeze([Y], [])
        Y1 = builder.aiOnnx.add([Ys, Y_h])
        Y2 = builder.aiOnnx.add([Y1, Y_c])
        builder.addOutputTensor(Y2)
        return [
            Y2,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + Y2
        ]

    def reference(ref_data):
        lstm = torch.nn.LSTM(input_size, hidden_size, 1)
        lstm.weight_ih_l0.data = torch.tensor(d2_torch[0])
        lstm.weight_hh_l0.data = torch.tensor(d3_torch[0])
        lstm.bias_ih_l0.data.fill_(0)
        lstm.bias_hh_l0.data.fill_(0)

        a = torch.tensor(d1, requires_grad=True)
        Y, (Y_h, Y_c) = lstm(a)
        Ys = Y.squeeze()
        Y1 = Ys + Y_h
        Y2 = Y1 + Y_c

        Y.retain_grad()
        Y_h.retain_grad()
        Y_c.retain_grad()
        Ys.retain_grad()
        Y1.retain_grad()

        d__o = ref_data.getOutputTensorGrad(0)
        Y2.backward(torch.tensor(d__o))

        # reorder the weights for comparison with popart
        wi, wf, wc, wo = torch.split(lstm.weight_ih_l0.grad, hidden_size)
        wig = torch.cat((wi, wo, wf, wc), dim=0)
        wig.unsqueeze_(0)

        # reorder the weights for comparison with popart
        wi, wf, wc, wo = torch.split(lstm.weight_hh_l0.grad, hidden_size)
        whg = torch.cat((wi, wo, wf, wc), dim=0)
        whg.unsqueeze_(0)

        return [Y2, a.grad, wig, whg, None]

    op_tester.passes = ['PreUniRepl']
    # relaxing the numerical precision required for this test:
    op_tester.atol = 1e-07
    op_tester.rtol = 1e-04
    op_tester.run(init_builder, reference, 'train')


def test_lstm_biases(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 5
    num_directions = 1

    d2 = np.random.rand(num_directions, 4 * hidden_size,
                        input_size).astype(np.float32)
    d3 = np.random.rand(num_directions, 4 * hidden_size,
                        hidden_size).astype(np.float32)
    d4 = np.random.rand(num_directions, 8 * hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3, i4], 3)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        lstm = LSTM_Helper(X=d1, W=d2, R=d3, B=d4)
        Y, Y_h, Y_c = lstm.step()

        return [Y, Y_h, Y_c]

    op_tester.run(init_builder, reference, 'infer')


def test_lstm_initial_hc(op_tester):
    d1 = np.array([[[1., 2., 3.]]]).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 2
    num_directions = 1

    d2 = np.random.rand(num_directions, 4 * hidden_size,
                        input_size).astype(np.float32)
    d3 = np.random.rand(num_directions, 4 * hidden_size,
                        hidden_size).astype(np.float32)
    d4 = np.random.rand(num_directions, 8 * hidden_size).astype(np.float32)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np.random.rand(num_directions, batch_size,
                               hidden_size).astype(np.float32)
    initial_c = np.random.rand(num_directions, batch_size,
                               hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(seq_lens)
        i6 = builder.addInputTensor(initial_h)
        i7 = builder.addInputTensor(initial_c)
        Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3, i4, i5, i6, i7], 3)
        builder.addOutputTensor(Y_h)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        lstm = LSTM_Helper(X=d1,
                           W=d2,
                           R=d3,
                           B=d4,
                           initial_h=initial_h,
                           initial_c=initial_c)
        Y, Y_h, Y_c = lstm.step()

        return [Y, Y_h, Y_c]

    op_tester.run(init_builder, reference, 'infer')


def test_lstm_torch_grad_all_inputs(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 2
    num_directions = 1

    wi = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wo = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wf = np.random.rand(1, hidden_size, input_size).astype(np.float32)
    wc = np.random.rand(1, hidden_size, input_size).astype(np.float32)

    whi = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    who = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whf = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)
    whc = np.random.rand(1, hidden_size, hidden_size).astype(np.float32)

    input_weights = np.concatenate((wi, wo, wf, wc), axis=1)
    input_weights_torch = np.concatenate((wi, wf, wc, wo), axis=1)

    hidden_weights = np.concatenate((whi, who, whf, whc), axis=1)
    hidden_weights_torch = np.concatenate((whi, whf, whc, who), axis=1)

    bii = np.random.rand(1, hidden_size).astype(np.float32)
    bio = np.random.rand(1, hidden_size).astype(np.float32)
    bif = np.random.rand(1, hidden_size).astype(np.float32)
    bic = np.random.rand(1, hidden_size).astype(np.float32)

    bhi = np.random.rand(1, hidden_size).astype(np.float32)
    bho = np.random.rand(1, hidden_size).astype(np.float32)
    bhf = np.random.rand(1, hidden_size).astype(np.float32)
    bhc = np.random.rand(1, hidden_size).astype(np.float32)

    biases = np.concatenate((bii, bio, bif, bic, bhi, bho, bhf, bhc), axis=1)
    input_biases_torch = np.concatenate((bii, bif, bic, bio), axis=1)
    hidden_biases_torch = np.concatenate((bhi, bhf, bhc, bho), axis=1)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np.random.rand(num_directions, batch_size,
                               hidden_size).astype(np.float32)
    initial_c = np.random.rand(num_directions, batch_size,
                               hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(input_weights)
        i3 = builder.addInputTensor(hidden_weights)
        i4 = builder.addInputTensor(biases)
        i5 = builder.addInputTensor(seq_lens)
        i6 = builder.addInputTensor(initial_h)
        i7 = builder.addInputTensor(initial_c)
        Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3, i4, i5, i6, i7], 3)
        Ys = builder.aiOnnx.squeeze([Y], [])
        Y1 = builder.aiOnnx.add([Ys, Y_h])
        Y2 = builder.aiOnnx.add([Y1, Y_c])
        builder.addOutputTensor(Y2)
        return [
            Y2,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + i4,
            popart.reservedGradientPrefix() + i6,
            popart.reservedGradientPrefix() + i7,
            popart.reservedGradientPrefix() + Y2
        ]

    def reference(ref_data):
        lstm = torch.nn.LSTM(input_size, hidden_size, 1)
        lstm.weight_ih_l0.data = torch.tensor(input_weights_torch[0])
        lstm.weight_hh_l0.data = torch.tensor(hidden_weights_torch[0])
        lstm.bias_ih_l0.data = torch.tensor(input_biases_torch)
        lstm.bias_hh_l0.data = torch.tensor(hidden_biases_torch)

        h0 = torch.tensor(initial_h, requires_grad=True)
        c0 = torch.tensor(initial_c, requires_grad=True)

        a = torch.tensor(d1, requires_grad=True)
        Y, (Y_h, Y_c) = lstm(a, (h0, c0))
        Ys = Y.squeeze()
        Y1 = Ys + Y_h
        Y2 = Y1 + Y_c

        Y.retain_grad()
        Y_h.retain_grad()
        Y_c.retain_grad()
        Ys.retain_grad()
        Y1.retain_grad()

        d__o = ref_data.getOutputTensorGrad(0)
        Y2.backward(torch.tensor(d__o))

        # reorder the weights for comparison with popart
        wi, wf, wc, wo = torch.split(lstm.weight_ih_l0.grad, hidden_size)
        wig = torch.cat((wi, wo, wf, wc), dim=0)
        wig.unsqueeze_(0)

        # reorder the weights for comparison with popart
        wi, wf, wc, wo = torch.split(lstm.weight_hh_l0.grad, hidden_size)
        whg = torch.cat((wi, wo, wf, wc), dim=0)
        whg.unsqueeze_(0)

        # reorder the biases for comparison with popart
        bii, bif, bic, bio = torch.split(lstm.bias_ih_l0.grad,
                                         hidden_size,
                                         dim=1)
        bhi, bhf, bhc, bho = torch.split(lstm.bias_hh_l0.grad,
                                         hidden_size,
                                         dim=1)
        b_grad = torch.cat((bii, bio, bif, bic, bhi, bho, bhf, bhc)).view(
            1, 8 * hidden_size)

        return [Y2, a.grad, wig, whg, b_grad, h0.grad, c0.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.atol = 1e-07
    op_tester.run(init_builder, reference, 'train')


class LSTM_Helper():
    def __init__(self, **params):  # type: (*Any) -> None
        # LSTM Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        C_0 = str('initial_c')
        P = str('P')
        number_of_gates = 4
        number_of_peepholes = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            b = params[B] if B in params else np.zeros(
                2 * number_of_gates * hidden_size, dtype=np.float32)
            p = params[P] if P in params else np.zeros(
                number_of_peepholes * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)
            c_0 = params[C_0] if C_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def h(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        [p_i, p_o, p_f] = np.split(self.P, 3)
        h_list = []
        H_t = self.H_0
        C_t = self.C_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, np.transpose(self.W)) + np.dot(
                H_t, np.transpose(self.R)) + np.add(*np.split(self.B, 2))
            i, o, f, c = np.split(gates, 4, -1)
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            h_list.append(H)
            H_t = H
            C_t = C
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1], C_t


@tu.requires_ipu_model
def test_import_torch_lstm(tmpdir):
    torch.manual_seed(0)
    np.random.seed(0)

    seq_length = 5
    batch_size = 2
    layers = 1

    # create an lstm module with defined input and hidden sizes
    def torch_create_lstm(input_size, hidden_size):
        class Module0(torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.lstm = torch.nn.LSTM(input_size, hidden_size, layers)

            def forward(self, inputs):
                x = self.lstm(inputs[0], inputs[1])
                return x

        return Module0()

    # export model created by `torch_create_lstm`
    def torch_export_lstm(onnx_file_name, model, inputs):
        print('pytorch exporting lstm')
        tt = [torch.tensor(i) for i in inputs]
        dummy_input = [tt[0], (tt[1], tt[2])]

        torch.onnx.export(model,
                          dummy_input,
                          onnx_file_name,
                          input_names=['X', 'initial_h', 'initial_c'],
                          output_names=['Y', 'Y_h', 'Y_c'])

    # create a random np array of shape `*shape` and type np.float32
    def np_rand(*shape):
        return np.random.rand(*shape).astype(np.float32)

    # run the torch lstm
    # also generate the onnx file
    def run_lstm_torch(input_size, hidden_size, onnx_file_name, inputs):
        # create torch lstm and export
        torch_lstm = torch_create_lstm(input_size, hidden_size)
        torch_export_lstm(onnx_file_name, torch_lstm, inputs)

        # run the torch session
        x = torch.tensor(inputs[0])
        h0 = torch.tensor(inputs[1])
        c0 = torch.tensor(inputs[2])
        out = torch_lstm.forward((x, (h0, c0)))
        return (out[0], out[1][0], out[1][1])

    def run_lstm_popart(onnx_file_name, inputs):
        # generate a popart session
        builder = popart.Builder(onnx_file_name)
        outputs = builder.getOutputTensorIds()
        dataFlow = popart.DataFlow(1, outputs)
        device = tu.create_test_device(1, 1216, opts={"tilesPerIPU": 1216})
        s = popart.InferenceSession(fnModel=onnx_file_name,
                                    dataFeed=dataFlow,
                                    deviceInfo=device)

        anchor_map = s.initAnchorArrays()
        s.prepareDevice()

        # run the popart session
        input_map = {
            'X': inputs[0],
            'initial_h': inputs[1],
            'initial_c': inputs[2]
        }
        stepio = popart.PyStepIO(input_map, anchor_map)
        s.run(stepio)

        return (anchor_map['Y'], anchor_map['Y_h'], anchor_map['Y_c'])

    input_size = 2
    hidden_size = 7
    fname = str(tmpdir / 'bar.onnx')

    # create inputs
    x = np_rand(seq_length, batch_size, input_size)
    h0 = np_rand(layers, batch_size, hidden_size)
    c0 = np_rand(layers, batch_size, hidden_size)

    torch_out = run_lstm_torch(input_size, hidden_size, fname, (x, h0, c0))
    popart_out = run_lstm_popart(fname, (x, h0, c0))

    # check the outputs
    assert len(popart_out) == 3 and len(torch_out) == 3

    for i, (po, to) in enumerate(zip(popart_out, torch_out)):
        print('Checking output {}'.format(i))
        assert np.allclose(po, to.data.numpy())


@tu.requires_ipu_model
def test_import_torch_lstm_train(tmpdir):
    torch.manual_seed(0)
    np.random.seed(0)

    seq_length = 5
    batch_size = 2
    layers = 1

    # create an lstm module with defined input and hidden sizes
    def torch_create_lstm(input_size, hidden_size):
        class Module0(torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.lstm = torch.nn.LSTM(input_size, hidden_size, layers)

            def forward(self, inputs):
                x = self.lstm(inputs[0], inputs[1])
                return x[0] + x[1][0] + x[1][1]

        return Module0()

    # export model created by `torch_create_lstm`
    def torch_export_lstm(onnx_file_name, model, inputs):
        print('pytorch exporting lstm')
        tt = [torch.tensor(i) for i in inputs]
        dummy_input = [tt[0], (tt[1], tt[2])]

        torch.onnx.export(model,
                          dummy_input,
                          onnx_file_name,
                          input_names=['X', 'initial_h', 'initial_c'],
                          output_names=['out'],
                          do_constant_folding=False)

    # create a random np array of shape `*shape` and type np.float32
    def np_rand(*shape):
        return np.random.rand(*shape).astype(np.float32)

    # run the torch lstm
    def run_lstm_torch(torch_lstm, inputs, d__out):
        # run the torch session
        x = torch.tensor(inputs[0], requires_grad=True)
        h0 = torch.tensor(inputs[1], requires_grad=True)
        c0 = torch.tensor(inputs[2], requires_grad=True)

        torch_lstm.lstm.weight_ih_l0.requires_grad_(True)
        torch_lstm.lstm.weight_hh_l0.requires_grad_(True)

        out = torch_lstm.forward((x, (h0, c0)))

        d__out = torch.tensor(d__out)
        out.backward(d__out)

        # manually update parameters
        for name, param in torch_lstm.named_parameters():
            print('Updating lstm param {}'.format(name))
            param.data.sub_(0.1 * param.grad.data)

        outputs = {
            'out':
            out,
            popart.reservedGradientPrefix() + 'X':
            x.grad,
            popart.reservedGradientPrefix() + 'initial_h':
            h0.grad,
            popart.reservedGradientPrefix() + 'initial_c':
            c0.grad,
            popart.reservedGradientPrefix() + 'W':
            torch_lstm.lstm.weight_ih_l0.grad,
            popart.reservedGradientPrefix() + 'R':
            torch_lstm.lstm.weight_hh_l0.grad,
            popart.reservedGradientPrefix() + 'WB':
            torch_lstm.lstm.bias_ih_l0.grad,
            popart.reservedGradientPrefix() + 'RB':
            torch_lstm.lstm.bias_hh_l0.grad,
        }
        return {key: value.data.numpy() for key, value in outputs.items()}

    def get_popart_fname(fname):
        path = Path(fname)
        path = path.parent / ('popart_' + path.name)
        return str(path)

    def get_torch_fname(fname):
        path = Path(fname)
        path = path.parent / ('torch_' + path.name)
        return str(path)

    def run_lstm_popart(onnx_file_name, inputs):
        # generate a popart session
        builder = popart.Builder(onnx_file_name)
        outputs = builder.getOutputTensorIds()
        anchors = outputs + [
            popart.reservedGradientPrefix() + 'out',
            popart.reservedGradientPrefix() + 'X',
            popart.reservedGradientPrefix() + 'initial_h',
            popart.reservedGradientPrefix() + 'initial_c',
            popart.reservedGradientPrefix() + 'lstm.weight_ih_l0',
            popart.reservedGradientPrefix() + 'lstm.weight_hh_l0',
            popart.reservedGradientPrefix() + 'lstm.bias_ih_l0',
            popart.reservedGradientPrefix() + 'lstm.bias_hh_l0'
        ]
        dataFlow = popart.DataFlow(1, anchors)
        optimizer = popart.ConstSGD(0.1)
        losses = [popart.L1Loss('out', "l1LossVal", 0.1)]
        device = tu.create_test_device(1, 1216, opts={"tilesPerIPU": 1216})
        print('Creating session')
        s = popart.TrainingSession(fnModel=onnx_file_name,
                                   dataFeed=dataFlow,
                                   optimizer=optimizer,
                                   losses=losses,
                                   passes=popart.Patterns(["PreUniRepl"]),
                                   deviceInfo=device)
        print('setting device')

        anchor_map = s.initAnchorArrays()
        s.prepareDevice()

        # run the popart session
        input_map = {
            'X': inputs[0],
            'initial_h': inputs[1],
            'initial_c': inputs[2]
        }
        stepio = popart.PyStepIO(input_map, anchor_map)
        s.weightsFromHost()
        s.run(stepio)
        s.modelToHost(get_popart_fname(onnx_file_name))

        anchor_map[popart.reservedGradientPrefix() +
                   'W'] = anchor_map.pop(popart.reservedGradientPrefix() +
                                         'lstm.weight_ih_l0')
        anchor_map[popart.reservedGradientPrefix() +
                   'R'] = anchor_map.pop(popart.reservedGradientPrefix() +
                                         'lstm.weight_hh_l0')
        anchor_map[popart.reservedGradientPrefix() +
                   'WB'] = anchor_map.pop(popart.reservedGradientPrefix() +
                                          'lstm.bias_ih_l0')
        anchor_map[popart.reservedGradientPrefix() +
                   'RB'] = anchor_map.pop(popart.reservedGradientPrefix() +
                                          'lstm.bias_hh_l0')
        return anchor_map

    input_size = 2
    hidden_size = 7
    fname = str(tmpdir / 'bar.onnx')

    # create inputs
    x = np_rand(seq_length, batch_size, input_size)
    h0 = np_rand(layers, batch_size, hidden_size)
    c0 = np_rand(layers, batch_size, hidden_size)

    torch_lstm = torch_create_lstm(input_size, hidden_size)
    torch_export_lstm(fname, torch_lstm, (x, h0, c0))
    popart_out = run_lstm_popart(fname, (x, h0, c0))
    torch_out = run_lstm_torch(
        torch_lstm, (x, h0, c0),
        popart_out.pop(popart.reservedGradientPrefix() + 'out'))
    torch_export_lstm(get_torch_fname(fname), torch_lstm, (x, h0, c0))

    nr = popart.NumericsReport(fname, get_torch_fname(fname), fname,
                               get_popart_fname(fname))
    print(nr.fullReport())

    assert len(popart_out.keys()) == 8
    assert len(popart_out.keys()) == len(torch_out.keys())

    errors = 0
    for key in popart_out.keys():
        po = popart_out[key]
        to = torch_out[key]
        print('Checking {}'.format(key))
        if po.shape != to.shape:
            errors += 1
            print('tensors {} are not matching shapes'.format(key))
            print()
        elif not np.allclose(po, to):
            errors += 1
            print('tensors {} are not close'.format(key))
            print('  popart')
            print('    {}'.format(po))
            print('  torch')
            print('    {}'.format(to))
            print()
    assert errors == 0


@tu.requires_ipu_model
def test_import_torch_lstm_multi_run(tmpdir):
    torch.manual_seed(0)
    np.random.seed(0)

    seq_length = 5
    batch_size = 2
    layers = 1

    # create an lstm module with defined input and hidden sizes
    def torch_create_lstm(input_size, hidden_size):
        class Module0(torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.lstm = torch.nn.LSTM(input_size, hidden_size, layers)

            def forward(self, inputs):
                x = self.lstm(inputs[0], inputs[1])
                return x

        return Module0()

    # export model created by `torch_create_lstm`
    def torch_export_lstm(onnx_file_name, model, inputs):
        print('pytorch exporting lstm')
        tt = [torch.tensor(i) for i in inputs]
        dummy_input = [tt[0], (tt[1], tt[2])]

        torch.onnx.export(model,
                          dummy_input,
                          onnx_file_name,
                          input_names=['X', 'initial_h', 'initial_c'],
                          output_names=['Y', 'Y_h', 'Y_c'])

    # create a random np array of shape `*shape` and type np.float32
    def np_rand(*shape):
        return np.random.rand(*shape).astype(np.float32)

    # run the torch lstm
    # also generate the onnx file
    def run_lstm_torch(input_size, hidden_size, onnx_file_name, inputs):
        export_inputs = [i for i in inputs]
        export_inputs[0] = export_inputs[0][0]
        export_inputs[0] = export_inputs[0].reshape(1, *export_inputs[0].shape)

        # create torch lstm and export
        torch_lstm = torch_create_lstm(input_size, hidden_size)
        torch_export_lstm(onnx_file_name, torch_lstm, export_inputs)

        # run the torch session
        x = torch.tensor(inputs[0])
        h0 = torch.tensor(inputs[1])
        c0 = torch.tensor(inputs[2])
        hidden = (h0, c0)

        # calculate the lstm step by step
        outs = []
        for i in range(x.shape[0]):
            i = x[i]
            i = i.view(1, *i.shape)
            (o, hidden) = torch_lstm.forward((i, hidden))
            outs.append(o)
        outs = torch.cat(outs)

        return (outs, hidden[0], hidden[1])

    def run_lstm_popart(onnx_file_name, inputs):
        # generate a popart session
        builder = popart.Builder(onnx_file_name)
        outputs = builder.getOutputTensorIds()
        dataFlow = popart.DataFlow(1, outputs)
        device = tu.create_test_device(1, 1216, opts={"tilesPerIPU": 1216})
        s = popart.InferenceSession(fnModel=onnx_file_name,
                                    dataFeed=dataFlow,
                                    deviceInfo=device)

        anchor_map = s.initAnchorArrays()
        s.prepareDevice()

        h0 = inputs[1]
        c0 = inputs[2]

        outs = []
        for i in range(inputs[0].shape[0]):
            input_data = inputs[0][i]
            input_data = input_data.reshape(1, *input_data.shape)
            input_map = {'X': input_data, 'initial_h': h0, 'initial_c': c0}
            stepio = popart.PyStepIO(input_map, anchor_map)
            s.run(stepio)

            h0 = anchor_map['Y_h']
            c0 = anchor_map['Y_c']
            outs.append(np.copy(anchor_map['Y']))

        outs = np.concatenate(outs)

        return (outs, anchor_map['Y_h'], anchor_map['Y_c'])

    input_size = 2
    hidden_size = 7
    fname = str(tmpdir / 'bar.onnx')

    # create inputs
    x = np_rand(seq_length, batch_size, input_size)
    h0 = np_rand(layers, batch_size, hidden_size)
    c0 = np_rand(layers, batch_size, hidden_size)

    torch_out = run_lstm_torch(input_size, hidden_size, fname, (x, h0, c0))
    popart_out = run_lstm_popart(fname, (x, h0, c0))

    # check the outputs
    assert len(popart_out) == 3 and len(torch_out) == 3

    for i, (po, to) in enumerate(zip(popart_out, torch_out)):
        print('Checking output {}'.format(i))
        assert np.allclose(po, to.data.numpy())


# This tests certain configurations of torch lstm that can
# export a model with a constant of shape operation.
def test_lstm_export_with_constantofshape(tmpdir):
    np.random.seed(42)
    torch.manual_seed(43)

    class RNNNet(torch.nn.Module):
        def __init__(self):
            super(RNNNet, self).__init__()

            hidden_size = 8
            input_size = 18

            self.lstm = torch.nn.LSTM(input_size=input_size,
                                      hidden_size=hidden_size,
                                      batch_first=True)

        def forward(self, x):
            x, (h, c) = self.lstm(x)
            return x

    net = RNNNet()
    np_data = np.random.rand(1, 100, 18).astype(np.float32)
    torch_data = torch.from_numpy(np_data)
    torchOutput = net(torch_data).detach().numpy()

    export_name = str(tmpdir / "lstm_small_repro.onnx")

    torch.onnx.export(net,
                      torch_data,
                      export_name,
                      verbose=True,
                      input_names=['data'],
                      output_names=['tag'])

    # Verify this model contains a ConstantOfShape op.
    model = onnx.load(export_name)
    nodes = model.graph.node
    nodes = [i for i in nodes if i.op_type == 'ConstantOfShape']
    assert len(nodes) > 0

    inputShapeInfo = popart.InputShapeInfo()
    inputShapeInfo.add("data", popart.TensorInfo("FLOAT", [1, 100, 18]))

    anchors = {"tag": popart.AnchorReturnType("All")}
    dataFeed = popart.DataFlow(1, anchors)
    device = tu.create_test_device()

    session = popart.InferenceSession(export_name,
                                      dataFeed,
                                      device,
                                      inputShapeInfo=inputShapeInfo)

    session.prepareDevice()

    inferenceAnchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({"data": np_data}, inferenceAnchors)
    session.run(stepio)
    popartOutput = inferenceAnchors['tag']

    assert torchOutput.shape == popartOutput.shape
    assert np.allclose(torchOutput, popartOutput, atol=1e-07)
