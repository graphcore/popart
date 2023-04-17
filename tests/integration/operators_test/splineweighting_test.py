# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, List

import torch
import pytest

from torch_spline_conv import spline_weighting as spline_weighting_ref


def splineweighting_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    basis: torch.Tensor,
    weight_index: torch.Tensor,
    dtype: torch.dtype,
) -> List[torch.Tensor]:

    output = spline_weighting_ref(input, weight, basis, weight_index)
    return [output.type(dtype)]


def splineweighting_builder(
    builder: Any,
    input: torch.Tensor,
    weight: torch.Tensor,
    basis: torch.Tensor,
    weight_index: torch.Tensor,
):
    P = builder.addInputTensor(input.numpy())
    W = builder.addInputTensor(weight.numpy())
    B = builder.addInputTensor(basis.numpy())
    WI = builder.addInputTensor(weight_index.numpy())
    input_tensors = [P, W, B, WI]

    output = builder.aiGraphcore.splineweighting(input_tensors)

    builder.addOutputTensor(output)

    return [output]


def splineweighting_op_test_body(op_tester, data, dtype):
    input_data, weight_data, basis_data, weight_index_data = data

    def init_builder(builder):
        input = input_data.type(dtype)
        weight = weight_data.type(dtype)
        basis = basis_data.type(dtype)
        weight_index = weight_index_data.type(torch.int32)
        return splineweighting_builder(builder, input, weight, basis, weight_index)

    def reference(_):
        input = input_data.type(torch.float)
        weight = weight_data.type(torch.float)
        basis = basis_data.type(torch.float)
        weight_index = weight_index_data.type(torch.long)
        return splineweighting_reference(input, weight, basis, weight_index, dtype)

    if dtype == torch.float16:
        # Adjust the tolerance since reference is computed on float32
        op_tester.atol = 1e-02
        op_tester.rtol = 1e-05

    op_tester.run(init_builder, reference)


def generate_input_data(num_edges, in_ch, out_ch, kernel_size, num_splines):
    torch.manual_seed(0)
    input = torch.rand(num_edges, in_ch)
    weights = torch.rand(kernel_size, in_ch, out_ch)
    basis = torch.rand(num_edges, num_splines)
    weight_index = torch.randint(0, kernel_size, (num_edges, num_splines))
    return input, weights, basis, weight_index


test_params = (
    {"num_edges": 6, "in_ch": 4, "out_ch": 4, "kernel_size": 10, "num_splines": 8},
    {"num_edges": 16, "in_ch": 5, "out_ch": 6, "kernel_size": 6, "num_splines": 6},
)


@pytest.mark.parametrize("params", test_params)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_basis(op_tester, params, dtype):
    data = generate_input_data(**params)
    splineweighting_op_test_body(op_tester, data, dtype)
