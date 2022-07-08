# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import test_util as tu

from loss_scaling_util_test import (
    getOptimizers,
    run_automatic_loss_scaling_comparison_test,
)


@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates(tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(tmpdir, optimizer)


@tu.requires_ipu_model
@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_sharded(tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(tmpdir, shard=True, optimizer=optimizer)
