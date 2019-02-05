import numpy as np
import torch
from op_tester import op_tester


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
        return [Y2, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + Y2]

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

        # reorder the weights for comparison with poponnx
        wi, wf, wc, wo = torch.split(lstm.weight_ih_l0.grad, hidden_size)
        wig = torch.cat((wi, wo, wf, wc), dim=0)
        wig.unsqueeze_(0)

        # reorder the weights for comparison with poponnx
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
        lstm = LSTM_Helper(
            X=d1, W=d2, R=d3, B=d4, initial_h=initial_h, initial_c=initial_c)
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
            Y2, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + i4, 'd__' + i6,
            'd__' + i7, 'd__' + Y2
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

        # reorder the weights for comparison with poponnx
        wi, wf, wc, wo = torch.split(lstm.weight_ih_l0.grad, hidden_size)
        wig = torch.cat((wi, wo, wf, wc), dim=0)
        wig.unsqueeze_(0)

        # reorder the weights for comparison with poponnx
        wi, wf, wc, wo = torch.split(lstm.weight_hh_l0.grad, hidden_size)
        whg = torch.cat((wi, wo, wf, wc), dim=0)
        whg.unsqueeze_(0)

        # reorder the biases for comparison with poponnx
        bii, bif, bic, bio = torch.split(
            lstm.bias_ih_l0.grad, hidden_size, dim=1)
        bhi, bhf, bhc, bho = torch.split(
            lstm.bias_hh_l0.grad, hidden_size, dim=1)
        b_grad = torch.cat((bii, bhi, bio, bho, bif, bhf, bic, bhc)).view(
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
