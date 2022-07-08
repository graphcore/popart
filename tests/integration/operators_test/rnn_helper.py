# This file contains helper functions for testing RNN, GRU and LSTM ops in PopART.
# These are taken from https://github.com/onnx/onnx/tree/rel-1.6.0.
# Code in this file is available under the following MIT License:

# MIT License

# Copyright (c) ONNX Project Contributors
# All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


# Taken from https://github.com/onnx/onnx/blob/rel-1.6.0/onnx/backend/test/case/node/rnn.py
class RNN_Helper:
    def __init__(self, **params):  # type: (**Any) -> None
        # RNN Input Names
        X = str("X")
        W = str("W")
        R = str("R")
        B = str("B")
        H_0 = str("initial_h")

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[str(W)].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            b = (
                params[B]
                if B in params
                else np.zeros(2 * hidden_size, dtype=np.float32)
            )
            h_0 = (
                params[H_0]
                if H_0 in params
                else np.zeros((batch_size, hidden_size), dtype=np.float32)
            )

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        h_list = []
        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            H = self.f(
                np.dot(x, np.transpose(self.W))
                + np.dot(H_t, np.transpose(self.R))
                + np.add(*np.split(self.B, 2))
            )
            h_list.append(H)
            H_t = H
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1]


# Taken from https://github.com/onnx/onnx/blob/rel-1.6.0/onnx/backend/test/case/node/gru.py
class GRU_Helper:
    def __init__(self, **params):  # type: (*Any) -> None
        # GRU Input Names
        X = str("X")
        W = str("W")
        R = str("R")
        B = str("B")
        H_0 = str("initial_h")
        LBR = str("linear_before_reset")
        number_of_gates = 3

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

            b = (
                params[B]
                if B in params
                else np.zeros(2 * number_of_gates * hidden_size)
            )
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size))
            lbr = params[LBR] if LBR in params else 0

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr

        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        h_list = []
        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
            z, r = np.split(gates, 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(
                np.dot(x, np.transpose(w_h))
                + np.dot(r * H_t, np.transpose(r_h))
                + w_bh
                + r_bh
            )
            h_linear = self.g(
                np.dot(x, np.transpose(w_h))
                + r * (np.dot(H_t, np.transpose(r_h)) + r_bh)
                + w_bh
            )
            h = h_linear if self.LBR else h_default
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1]


# Based on LSTM_Helper from https://github.com/onnx/onnx/blob/rel-1.6.0/onnx/backend/test/case/node/lstm.py
class LSTM_Helper:
    def __init__(self, **params):  # type: (*Any) -> None
        # LSTM Input Names
        X = str("X")
        W = str("W")
        R = str("R")
        B = str("B")
        H_0 = str("initial_h")
        C_0 = str("initial_c")
        P = str("P")
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

            b = (
                params[B]
                if B in params
                else np.zeros(2 * number_of_gates * hidden_size, dtype=np.float32)
            )
            p = (
                params[P]
                if P in params
                else np.zeros(number_of_peepholes * hidden_size, dtype=np.float32)
            )
            h_0 = (
                params[H_0]
                if H_0 in params
                else np.zeros((batch_size, hidden_size), dtype=np.float32)
            )
            c_0 = (
                params[C_0]
                if C_0 in params
                else np.zeros((batch_size, hidden_size), dtype=np.float32)
            )

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
            gates = (
                np.dot(x, np.transpose(self.W))
                + np.dot(H_t, np.transpose(self.R))
                + np.add(*np.split(self.B, 2))
            )
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
