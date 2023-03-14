# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, List

import torch
import pytest


def bucketize_reference(
    input: torch.Tensor,
    boundaries: torch.Tensor,
    right: bool = False,
) -> List[torch.Tensor]:

    return [torch.bucketize(input, boundaries, out_int32=True, right=right)]


def bucketize_builder(
    builder: Any,
    input: torch.Tensor,
    boundaries: torch.Tensor,
    right: bool = False,
):
    I = builder.addInputTensor(input.numpy())
    B = builder.addInputTensor(boundaries.numpy())
    input_tensors = [I, B]

    out = builder.aiGraphcore.bucketize(
        input_tensors,
        right=int(right),
    )

    builder.addOutputTensor(out)

    return [out]


def bucketize_op_test_body(op_tester, right, data, dtype):
    input_data, boundaries_data = data
    input = torch.tensor(input_data, dtype=dtype)
    boundaries = torch.tensor(boundaries_data, dtype=dtype)

    def init_builder(builder):
        return bucketize_builder(builder, input, boundaries, right)

    def reference(_):
        return bucketize_reference(input, boundaries, right)

    op_tester.run(init_builder, reference)


basic_test_data = [
    ([[3, 6, 9], [3, 6, 10], [-1, 0, 1], [8, 9, 140]], [1, 3, 5, 7, 9]),
    ([[2, 5, 10], [6, 8, 3]], [1, 5, 7, 8, 10]),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6]),
    ([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], [1, 2, 3, 4, 5, 6]),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 6, 4, 5]),
]


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("data", basic_test_data)
def test_basic(op_tester, right, data, dtype):
    bucketize_op_test_body(op_tester, right, data, dtype)


fp_test_data = [
    ([1, 2, 3, 4, 5, 6, 7, 8, 9], [0.9, 1, 2, 2, 3, 3, 4, 4.1, 9, 9]),
    (
        [[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]],
        [0.9, 1, 2, 2, 3, 3, 4, 4.1, 9, 9],
    ),
]


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("data", basic_test_data)
def test_fp(op_tester, right, data, dtype):
    bucketize_op_test_body(op_tester, right, data, dtype)


@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("unrepresentable_value", ["nan", "inf"])
def test_special_case_fp(op_tester, right, unrepresentable_value):
    data = (
        [1.0, float(unrepresentable_value), 2.0, float(unrepresentable_value)],
        [0.0, 1.0, 2.0, 3.0],
    )
    bucketize_op_test_body(op_tester, right, data, torch.float32)


@pytest.mark.parametrize("right", [True, False])
def test_scalar(op_tester, right):
    data = ([1.0, float("nan"), 2.0, float("nan")], [1.0])
    bucketize_op_test_body(op_tester, right, data, torch.float32)
