# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester


def test_softsign(op_tester):
    """Test the softsign operator."""
    # Make the test on 250 random numbers
    input_data = np.random.rand(1, 250).astype(np.float32)

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
        torch_softsign = torch.nn.Softsign()
        result = [torch_softsign(torch_test_data)]
        return result

    op_tester.run(init_builder, reference, 'infer')
    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_softsign_grad(op_tester):
    """Test the softsign gradient operator."""
    input_data = np.random.rand(1, 250).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.softsign([i1])
        builder.addOutputTensor(o)
        op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
        result = [o, popart.reservedGradientPrefix() + i1]
        return result

    def reference(_):
        """Return the result of the gradient of the PyTorch softsign function."""
        torch_test_data = torch.tensor(input_data, requires_grad=True)
        torch_softsign = torch.nn.Softsign()
        fwd = torch_softsign(torch_test_data)
        # NOTE: We set the gradient w.r.t. to one in order to get a valid result
        bwd = fwd.backward(torch.ones(fwd.shape))
        result = [fwd, bwd]
        return result

    op_tester.run(init_builder, reference, 'train')
