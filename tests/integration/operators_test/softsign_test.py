# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import torch.nn.functional as F
from op_tester import op_tester


def create_input():
    # Make the test on 250 numbers over a wide range of values to test saturation behaviour
    f32info = np.finfo(np.float32)
    return np.linspace(f32info.min, f32info.max, num=250, dtype=np.float32)


def test_softsign(op_tester):
    """Test the softsign operator."""
    # Make the test on 250 random numbers
    input_data = create_input()

    def init_builder(builder):
        """Build the softsign operator."""
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.softsign([i1])
        builder.addOutputTensor(o)
        result = [o]
        return result

    def reference(_):
        """Return the result of the PyTorch softsign function."""
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        result = [F.softsign(torch_test_data)]
        return result

    op_tester.run(init_builder, reference, 'infer')


def test_softsign_grad(op_tester):
    """Test the softsign gradient operator."""
    input_data = create_input()

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.softsign([i1])
        builder.addOutputTensor(o)
        op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
        # Set the result to
        # ['Softsign:0', 'Gradient___input', 'Gradient___Softsign:0']
        result = [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]
        return result

    def reference(ref_data):
        """Return the result of the gradient of the PyTorch softsign function."""
        torch_test_data = torch.tensor(input_data, requires_grad=True)
        fwd = F.softsign(torch_test_data)
        # Return the gradient of the input tensor at index 0, i.e. the gradient of softplus
        d__o = ref_data.getOutputTensorGrad(0)
        # Set the gradient to torch_test_data.grad
        fwd.backward(torch.tensor(d__o))
        # Set the result so that it will correspond to
        # ['Softsign:0', 'Gradient___input', 'Gradient___Softsign:0']
        result = [fwd, torch_test_data.grad, d__o]
        return result

    op_tester.run(init_builder, reference, 'train')
