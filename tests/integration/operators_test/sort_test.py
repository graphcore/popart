# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, List

import torch
import pytest

import popart


def sort_reference(
    input: torch.Tensor,
    dim: int = -1,
    descending: bool = False,
    stable: bool = False,
    backward: bool = False,
    ref_data: Any = None,
) -> List[torch.Tensor]:

    if backward:
        input = input.clone()
        input.requires_grad_()

    values, indices = torch.sort(input, dim=dim, descending=descending, stable=stable)

    if backward:
        d__o = ref_data.getOutputTensorGrad(0)
        values.backward(torch.tensor(d__o))
        return [values, indices.int(), input.grad, None]

    return [values, indices.int()]


def sort_builder(
    builder: Any,
    input: torch.Tensor,
    dim: int = -1,
    descending: bool = False,
    stable: bool = False,
    backward: bool = False,
):
    I = builder.addInputTensor(input.numpy())
    input_tensors = [I]

    values, indices = builder.aiGraphcore.sort(
        input_tensors, axis=dim, descending=int(descending), stable=int(stable)
    )

    builder.addOutputTensor(values)
    builder.addOutputTensor(indices)

    result = [values, indices]

    if backward:
        result.append(popart.reservedGradientPrefix() + I)
        result.append(popart.reservedGradientPrefix() + values)

    return result


@pytest.mark.parametrize("shape", [(17, 4), (18, 10, 5)])
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("backward", [True, False])
def test_sort_basic(op_tester, shape, descending, backward):
    torch.manual_seed(42)
    input = torch.randn(*shape)

    for dim in range(input.ndim):

        def init_builder(builder):
            return sort_builder(builder, input, dim, descending, backward=backward)

        def reference(ref_data):
            return sort_reference(
                input, dim, descending, backward=backward, ref_data=ref_data
            )

        op_tester.run(init_builder, reference, "train" if backward else "infer")


@pytest.mark.parametrize("descending", [True, False])
def test_sort_stable(op_tester, descending):
    torch.manual_seed(42)
    input = torch.tensor([[2.0, 2.0, 1.0, 10.0, 11.0], [2.0, 15.0, 15.0, 10.0, 11.0]])

    for dim in range(input.ndim):

        def init_builder(builder):
            return sort_builder(builder, input, dim, descending, stable=True)

        def reference(_):
            return sort_reference(input, dim, descending, stable=True)

        op_tester.run(init_builder, reference)
