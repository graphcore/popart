# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import test_util as tu

from loss_scaling_util_test import getModelProto, compare_weights, run_automatic_loss_scaling_comparison_test, getOptimizers


@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_update_period(
        tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(
        tmpdir,
        optimizer,
        bps=8,
        update_period=1,
        expected_loss_scale=[10., 20., 40., 80., 160., 320., 640., 1280.])


# Someone would expect expected_loss_scale to begin with 20 and not 10 as
# we have in this test check. The reason is that the topology of the graph
# is such that MulOp(inputs: lossScaleUpdateFactor, lossScaling)
# is executed before LossScaleUpdateOp.
# See automaticlossscaling.hpp documentation.


@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_update_period2(
        tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(
        tmpdir,
        optimizer,
        bps=8,
        update_period=2,
        expected_loss_scale=[10., 20., 20., 40., 40., 80., 80., 160.])


@tu.requires_ipu_model
@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_sharded_update_period(
        tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(
        tmpdir,
        bps=8,
        shard=True,
        optimizer=optimizer,
        update_period=1,
        expected_loss_scale=[10., 20., 40., 80., 160., 320., 640., 1280.])


@tu.requires_ipu_model
@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_sharded_update_period2(
        tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(
        tmpdir,
        bps=8,
        shard=True,
        optimizer=optimizer,
        update_period=2,
        expected_loss_scale=[10., 20., 20., 40., 40., 80., 80., 160.])
