# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import test_util as tu

from loss_scaling_util_test import getModelProto, getOptimizers, run_automatic_loss_scaling_comparison_test


@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_grad_accumulation(
        tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(tmpdir,
                                               grad_accumulate=True,
                                               optimizer=optimizer)


@tu.requires_ipu_model
@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_grad_accumulation_shard(
        tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(tmpdir,
                                               grad_accumulate=True,
                                               shard=True,
                                               optimizer=optimizer)


@tu.requires_ipu_model
@pytest.mark.parametrize("optimizer", getOptimizers())
def test_auto_loss_scaling_identical_weight_updates_grad_accumulation_pipeline(
        tmpdir, optimizer):
    run_automatic_loss_scaling_comparison_test(tmpdir,
                                               grad_accumulate=True,
                                               shard=True,
                                               pipeline=True,
                                               optimizer=optimizer)
