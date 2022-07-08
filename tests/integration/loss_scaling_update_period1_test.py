# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import test_util as tu

from loss_scaling_util_test import (
    getOptimizers,
    run_automatic_loss_scaling_comparison_test,
)


@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_grad_accumulation_update_period(
    tmpdir, optimizer
):
    run_automatic_loss_scaling_comparison_test(
        tmpdir,
        bps=8,
        grad_accumulate=True,
        accumulation_factor=3,
        optimizer=optimizer,
        update_period=1,
        expected_loss_scale=[
            10.0,
            10.0,
            10.0,
            20.0,
            20.0,
            20.0,
            40.0,
            40.0,
            40.0,
            80.0,
            80.0,
            80.0,
            160.0,
            160.0,
            160.0,
            320.0,
            320.0,
            320.0,
            640.0,
            640.0,
            640.0,
            1280.0,
            1280.0,
            1280.0,
        ],
    )


@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_grad_accumulation_update_period2(
    tmpdir, optimizer
):
    run_automatic_loss_scaling_comparison_test(
        tmpdir,
        bps=8,
        grad_accumulate=True,
        accumulation_factor=3,
        optimizer=optimizer,
        update_period=2,
        expected_loss_scale=[
            10.0,
            10.0,
            10.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            160.0,
            160.0,
            160.0,
        ],
    )


@tu.requires_ipu_model
@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_grad_accumulation_pipeline_update_period(
    tmpdir, optimizer
):
    run_automatic_loss_scaling_comparison_test(
        tmpdir,
        bps=8,
        grad_accumulate=True,
        shard=True,
        pipeline=True,
        optimizer=optimizer,
        update_period=1,
        expected_loss_scale=[
            10.0,
            10.0,
            10.0,
            20.0,
            20.0,
            20.0,
            40.0,
            40.0,
            40.0,
            80.0,
            80.0,
            80.0,
            160.0,
            160.0,
            160.0,
            320.0,
            320.0,
            320.0,
            640.0,
            640.0,
            640.0,
            1280.0,
            1280.0,
            1280.0,
        ],
    )


@pytest.mark.skip(reason="TODO: T61577")
@tu.requires_ipu_model
@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_grad_accumulation_pipeline_update_period2(
    tmpdir, optimizer
):
    run_automatic_loss_scaling_comparison_test(
        tmpdir,
        bps=8,
        grad_accumulate=True,
        shard=True,
        pipeline=True,
        optimizer=optimizer,
        update_period=2,
        expected_loss_scale=[
            10.0,
            10.0,
            10.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            160.0,
            160.0,
            160.0,
        ],
    )
