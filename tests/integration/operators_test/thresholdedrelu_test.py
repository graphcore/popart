# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester


def thresholdedrelu_def(input, alpha):
    # Def: x <= alpha ? 0 : x
    theta = torch.ones(input.shape)
    theta[input <= alpha] = 0
    return input * theta


def test_thresholdedrelu(op_tester):
    input_data = np.linspace(-10, 10, 250, dtype=np.float32)
    alpha = 2.0

    def get_init_builder(builder_settings='NoInPlace'):
        def init_builder(builder):
            i1 = builder.addInputTensor(input_data)
            o = builder.aiOnnx.thresholdedrelu([i1], alpha=alpha)
            builder.addOutputTensor(o)
            result = [o]
            if builder_settings is 'InPlace':
                op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
            elif builder_settings is 'backward':
                result = [
                    o,
                    popart.TensorId(popart.reservedGradientPrefix() + i1),
                    popart.TensorId(popart.reservedGradientPrefix() + o)
                ]
            return result

        return init_builder

    def torch_calc(setting='forward'):
        def reference(ref_data):
            torch_test_data = torch.tensor(
                input_data,
                requires_grad=False if setting is 'forward' else True)
            torch_thresholdedrelu = lambda x: thresholdedrelu_def(x, alpha)
            m = torch_thresholdedrelu(torch_test_data)
            if setting is 'backward':
                d__o = ref_data.getOutputTensorGrad(0)
                m.backward(torch.tensor(d__o))
                result = [m, torch_test_data.grad, None]
            else:
                result = [m]
            return result

        return reference

    op_tester.run(get_init_builder(), torch_calc(), 'infer')
    op_tester.run(get_init_builder('InPlace'), torch_calc(), 'infer')
    op_tester.run(get_init_builder('backward'), torch_calc('backward'),
                  'train')
