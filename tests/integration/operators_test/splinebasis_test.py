# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, List

import torch
import pytest

from torch_spline_conv import spline_basis as spline_basis_ref


def splinebasis_reference(
    pseudo: torch.Tensor,
    kernel_size: torch.Tensor,
    is_open_spline: torch.Tensor,
    degree: int,
    dtype: torch.dtype,
) -> List[torch.Tensor]:

    basis, weight_index = spline_basis_ref(pseudo, kernel_size, is_open_spline, degree)

    return basis.type(dtype), weight_index.type(torch.int32)


def splinebasis_builder(
    builder: Any,
    pseudo: torch.Tensor,
    kernel_size: torch.Tensor,
    is_open_spline: torch.Tensor,
    degree: int,
):
    p = builder.addInputTensor(pseudo.numpy())
    ks = builder.addInputTensor(kernel_size.numpy())
    ios = builder.addInputTensor(is_open_spline.numpy())
    input_tensors = [p, ks, ios]

    [basis, weight_index] = builder.aiGraphcore.splinebasis(
        input_tensors,
        degree=degree,
    )

    builder.addOutputTensor(basis)
    builder.addOutputTensor(weight_index)

    return [basis, weight_index]


def splinebasis_op_test_body(op_tester, degree, data, dtype):
    psuedo_data, kernel_size_data, is_open_spline_data = data

    def init_builder(builder):
        pseudo = psuedo_data.type(dtype)
        kernel_size = kernel_size_data.type(torch.int32)
        is_open_spline = is_open_spline_data.type(torch.uint8)
        return splinebasis_builder(builder, pseudo, kernel_size, is_open_spline, degree)

    def reference(_):
        pseudo = psuedo_data.type(torch.float)
        kernel_size = kernel_size_data.type(torch.long)
        is_open_spline = is_open_spline_data.type(torch.uint8)
        return splinebasis_reference(pseudo, kernel_size, is_open_spline, degree, dtype)

    if dtype == torch.float16:
        # Adjust the tolerance since reference is computed on float32
        op_tester.atol = 1e-02
        op_tester.rtol = 1e-05
    op_tester.run(init_builder, reference)


def generate_input_data(num_edges, num_dims, maxKernelSize):
    torch.manual_seed(0)
    pseudo = torch.rand(num_edges, num_dims)
    kernel_size = torch.randint(1, maxKernelSize, (num_dims,))
    is_open_spline = torch.randint(0, 2, (num_dims,))
    return pseudo, kernel_size, is_open_spline


test_params = (
    {
        "num_edges": 6,
        "num_dims": 2,
        "maxKernelSize": 6,
    },
    {
        "num_edges": 64,
        "num_dims": 3,
        "maxKernelSize": 10,
    },
)


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("params", test_params)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_basis(op_tester, degree, params, dtype):
    data = generate_input_data(**params)
    splinebasis_op_test_body(op_tester, degree, data, dtype)
