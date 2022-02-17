# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import numpy as np
from popart.ir.random_seed import uint64_to_two_uint32, two_uint32_to_uint64, MAX_UINT64, create_seeds


def test_two_uint32_packing_and_unpacking():
    a, b = uint64_to_two_uint32(MAX_UINT64)
    num_out = two_uint32_to_uint64(a, b)
    assert MAX_UINT64 == num_out

    a, b = uint64_to_two_uint32(0)
    num_out = two_uint32_to_uint64(a, b)
    assert num_out == 0

    a, b = uint64_to_two_uint32(100)
    num_out = two_uint32_to_uint64(a, b)
    assert num_out == 100

    num = two_uint32_to_uint64(100, 32)
    a_out, b_out = uint64_to_two_uint32(num)
    assert a_out == 100
    assert b_out == 32

    num = two_uint32_to_uint64(31, 32)
    a_out, b_out = uint64_to_two_uint32(num)
    assert a_out == 31
    assert b_out == 32

    with pytest.raises(ValueError):
        uint64_to_two_uint32(-42)

    with pytest.raises(ValueError):
        uint64_to_two_uint32(MAX_UINT64 + 1)


def test_create_seeds():
    seed_tensors = create_seeds(42)
    assert str(seed_tensors.dtype) == 'uint32'
    assert seed_tensors.shape == (2, )

    seed_tensors_again = create_seeds(42)
    np.testing.assert_equal(seed_tensors, seed_tensors_again)

    seed_tensors = create_seeds(42,
                                batches_per_step=3,
                                gradient_accumulation_factor=7,
                                replicas=9)
    assert str(seed_tensors.dtype) == 'uint32'
    assert seed_tensors.shape == (3, 7, 9, 2)
