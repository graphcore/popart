# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester


def hardsigmoid_torch(input, alpha, beta):
    result = torch.max(
        torch.zeros(input.shape),
        torch.min(torch.ones(input.shape), alpha * input + beta))
    return result


def test_hardsigmoid(op_tester):
    input_data = np.linspace(-1, 2, 250, dtype=np.float32)
    alpha = 1.0
    beta = 0.5

    def get_init_builder(builder_settings='NoInPlace'):
        def init_builder(builder):
            i1 = builder.addInputTensor(input_data)
            o = builder.aiOnnx.hardsigmoid([i1], alpha=alpha, beta=beta)
            builder.addOutputTensor(o)
            result = [o]
            if builder_settings is 'InPlace':
                op_tester.passes = ['InPlace']
            elif builder_settings is 'backward':
                result = [
                    o,
                    popart.reservedGradientPrefix() + i1,
                    popart.reservedGradientPrefix() + o
                ]
            return result

        return init_builder

    def torch_calc(setting='forward'):
        def reference(ref_data):
            torch_test_data = torch.tensor(
                input_data,
                requires_grad=False if setting is 'forward' else True)
            m = hardsigmoid_torch(torch_test_data, alpha, beta)
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
