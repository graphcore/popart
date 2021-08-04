# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import onnx
from onnx import numpy_helper
import pytest
import test_util as tu

from loss_scaling_test import run_automatic_loss_scaling_comparison_test


@tu.requires_ipu
@pytest.mark.skip(
    "T31696: replicated tests are flaky due to some issue lower down the stack"
)
def test_auto_loss_scaling_identical_weight_updates_replicated(tmpdir):
    run_automatic_loss_scaling_comparison_test(tmpdir, replicate=True)


@tu.requires_ipu
@pytest.mark.skip(
    "T31696: replicated tests are flaky due to some issue lower down the stack"
)
def test_auto_loss_scaling_identical_weight_updates_replicated(tmpdir):
    run_automatic_loss_scaling_comparison_test(tmpdir,
                                               replicate=True,
                                               grad_accumulate=True)
