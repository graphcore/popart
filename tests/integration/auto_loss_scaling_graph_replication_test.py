# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import test_util as tu

from loss_scaling_test import run_automatic_loss_scaling_comparison_test


@tu.requires_ipu
@pytest.mark.skip(
    "TODO T69723: replicated tests are flaky due to some issue lower down the stack"
)
@pytest.mark.parametrize("grad_accumulate", (False, True))
def test_auto_loss_scaling_identical_weight_updates_replicated(tmpdir, grad_accumulate):
    run_automatic_loss_scaling_comparison_test(
        tmpdir, replicate=True, grad_accumulate=grad_accumulate
    )
