# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, List

import torch
import pytest


def nearbyint_reference(
    input: torch.Tensor,
) -> List[torch.Tensor]:

    return [torch.round(input)]


def nearbyint_builder(
    builder: Any,
    input: torch.Tensor,
) -> List[torch.Tensor]:
    I = builder.addInputTensor(input.numpy())
    input_tensors = [I]

    out = builder.aiGraphcore.nearbyint(
        input_tensors,
    )

    builder.addOutputTensor(out)

    return [out]


def nearbyint_op_test_body(op_tester, input_data, dtype):
    input = torch.tensor(input_data, dtype=dtype)

    def init_builder(builder):
        return nearbyint_builder(builder, input)

    def reference(_):
        return nearbyint_reference(input)

    op_tester.run(init_builder, reference)


basic_test_data = [
    [[3.10, 6.423, 9.123], [3.5, 6.0, 10.11], [-1.51, 0.49, 1.51], [8.50, 9.51, 140]],
    [[1, 2.5, 3.43, 4.5, 5, 6.5, 7.4324, 8.654, 9.532]],
    [1.24, 2.423, 3.243, 4.50, 5.50, 6.50, 7.90, 8.40, 9.59],
]


@pytest.mark.parametrize("data", basic_test_data)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_nearbyint(op_tester, data, dtype):
    nearbyint_op_test_body(op_tester, data, dtype)
