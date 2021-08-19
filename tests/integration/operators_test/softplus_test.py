# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester


def softplus_torch(input):
    """Return softplus = log(exp(x) + 1)"""
    # NOTE: The softplus implementation in PyTorch has two parameters
    #       beta and treshold which is not present in the ONNX
    #       specification
    result = torch.log(torch.exp(input) + torch.ones(input.shape))
    return result


def test_softplus(op_tester):
    """Test the softplus operator."""
    # Make the test on 250 random numbers
    input_data = np.random.rand(1, 250).astype(np.float32)

    def init_builder(builder):
        """Build the softplus operator."""
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.softplus([i1])
        builder.addOutputTensor(o)
        result = [o]
        return result

    def reference(_):
        """Return the result of the python softplus function above."""
        torch_test_data = torch.tensor(input_data, requires_grad=False)
        result = [softplus_torch(torch_test_data)]
        return result

    op_tester.run(init_builder, reference, 'infer')
    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_softplus_grad(op_tester):
    """Test the softplus gradient operator."""
    input_data = np.random.rand(1, 250).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.softplus([i1])
        builder.addOutputTensor(o)
        op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
        result = [o, popart.reservedGradientPrefix() + i1]
        return result

    def reference(_):
        """Return the result of the gradient of the python softplus function above."""
        torch_test_data = torch.tensor(input_data, requires_grad=True)
        fwd = softplus_torch(torch_test_data)
        # NOTE: We set the gradient w.r.t. to one in order to get a valid result
        bwd = fwd.backward(torch.ones(fwd.shape))
        result = [fwd, bwd]
        return result

    op_tester.run(init_builder, reference, 'train')
