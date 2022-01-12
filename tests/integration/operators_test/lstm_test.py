# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import popart
import torch
import json

from pathlib import Path
from test_session import PopartTestSession

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

# TODO: T19540 we had to split this test in 2 pieces to keep compile below 60s.
# This may have been due to a compile time regression. Investigate.


def np_rand(*shape):
    return np.random.rand(*shape).astype(np.float32)


def np_zeros(*shape):
    return np.zeros(shape, dtype=np.float32)


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
        i1 = builder.addInputTensor(d1, "input_data")
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
    op_tester.setPatterns(['LSTMOp', 'SplitGradOpToConcat'],
                          enableRuntimeAsserts=False)
    session = op_tester.run(init_builder, reference, 'train')

    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    graph = ir['maingraph']

    # There should be one lstm and it should be the aigraphcore lstm
    lstms = [op for op in graph if op['type'] == 'LSTM']
    assert len(lstms) == 1


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
    op_tester.setPatterns(['LSTMOp', 'SplitGradOpToConcat'],
                          enableRuntimeAsserts=False)
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
                [Y, Y_c] = output_ids
                assert builder.getTensorDtypeString(Y) == "float32"
                assert builder.getTensorDtypeString(Y_c) == "float32"

                assert builder.getTensorShape(Y) == [
                    inputs[0].shape[0], inputs[0].shape[1], inputs[1].shape[2]
                ]
                assert builder.getTensorShape(Y_c) == [
                    inputs[0].shape[1], inputs[1].shape[2]
                ]

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

            # NB shape inference is not yet possible with aiOnnx.split

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
            [Y, Y_c] = output_ids
            assert builder.getTensorDtypeString(Y) == "float32"
            assert builder.getTensorDtypeString(Y_c) == "float32"

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
    seq_lens_id = 'seqLen'

    def run_onnx_lstm(data, input_weights, output_weights, biases, seq_lens,
                      initial_h, initial_c):
        def init_builder(builder):
            tData = builder.addInputTensor(data, data_id)
            tIW = builder.addInitializedInputTensor(input_weights,
                                                    input_weights_id)
            tOW = builder.addInitializedInputTensor(output_weights,
                                                    output_weights_id)
            tBiases = builder.addInitializedInputTensor(biases, biases_id)
            tSeqLens = builder.addInputTensor(seq_lens, seq_lens_id)
            tInitH = builder.addInputTensor(initial_h, init_h_id)
            tInitC = builder.addInputTensor(initial_c, init_c_id)
            Y, Y_h, Y_c = builder.aiOnnx.lstm(
                [tData, tIW, tOW, tBiases, tSeqLens, tInitH, tInitC],
                3,
                clip=None)
            out = Y
            loss = builder.aiGraphcore.identityloss([out])

            return [
                loss,
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
                        initial_c):
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

            # NB shape inference is not yet possible with aiOnnx.split

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
            assert builder.getTensorDtypeString(out) == "float32"
            assert builder.getTensorDtypeString(cell_state) == "float32"

            loss = builder.aiGraphcore.identityloss([out])

            return [
                loss,
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

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32).squeeze()

    initial_h = np_rand(num_directions, batch_size, hidden_size)
    initial_c = np_rand(num_directions, batch_size, hidden_size)

    onnx_out = run_onnx_lstm(data, onnx_input_weights, onnx_output_weights,
                             onnx_biases, seq_lens, initial_h, initial_c)
    popart_out = run_popart_lstm(data, onnx_input_weights, onnx_output_weights,
                                 onnx_biases, initial_h, initial_c)

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

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
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

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    # relaxing the numerical precision required for this test:
    op_tester.atol = 1e-06
    op_tester.rtol = 1e-03
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
        i1 = builder.addInputTensor(d1, "input_1")
        i2 = builder.addInputTensor(d2, "input_2")
        i3 = builder.addInputTensor(d3, "input_3")
        i4 = builder.addInputTensor(d4, "input_4")
        i5 = builder.addInputTensor(seq_lens, "seq_lens")
        i6 = builder.addInputTensor(initial_h, "initial_h")
        i7 = builder.addInputTensor(initial_c, "initial_c")
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


def test_unsupported_activation(op_tester):
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
        builder.addNodeAttribute("activations", ["Affine", "Affine", "Affine"],
                                 {Y, Y_h, Y_c})
        builder.addOutputTensor(Y_h)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        # The reference should never run, popart should raise an exception before this.
        assert False

    op_tester.device = tu.create_test_device()
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert 'Affine' in e_info.value.args[0]
    assert 'not supported' in e_info.value.args[0]


# This tests that setup behaves in a predictable way for cloning ops.
# At the time of writing, this test failed with a segfault. This was
# due to LSTMOp::setup on a cloned lstm op, overwriting outputs that
# were connected to it during the explicit recompute transform.
def test_lstm_explicit_recompute():
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 7

    d2 = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, 4 * hidden_size, hidden_size)).astype(np.float32)

    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1))
    i2 = builder.addInitializedInputTensor(d2)
    i3 = builder.addInitializedInputTensor(d3)
    Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3], 3, clip=None)
    builder.recomputeOutputInBackwardPass(set([Y, Y_h, Y_c]))
    out = builder.aiOnnx.relu([Y])
    loss = builder.aiGraphcore.identityloss([out])

    builder.addOutputTensor(loss)

    device = tu.create_test_device()

    opts = popart.SessionOptions()
    opts.explicitRecomputation = True

    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
        deviceInfo=device,
        optimizer=popart.ConstSGD(0.1),
        userOptions=opts,
        loss=loss)

    # Now the test is passing we can add a check to make sure there
    # are 2 lstm ops. This is to check that the lstm is being cloned
    # and we are actually testing what we intended.
    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    maingraph = ir['maingraph']
    lstms = [i for i in maingraph if i['type'] == 'LSTM']
    assert len(lstms) == 2
