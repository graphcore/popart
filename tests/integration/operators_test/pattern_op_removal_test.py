# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
from op_tester import op_tester


def test_atan2_arg0_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns(['Atan2Arg1GradOp'], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiGraphcore.atan2([w1, i1], "test_atan2")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern Atan2Arg0GradOp"))


def test_atan2_arg1_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns(['Atan2Arg0GradOp'], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiGraphcore.atan2([i1, w1], "test_atan2")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern Atan2Arg1GradOp"))


def test_convflipweights_grad_error(op_tester):
    x = np.random.rand(1, 1, 2, 2).astype(np.float32)
    W = np.random.rand(1, 1, 2, 2).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInitializedInputTensor(W)

        o = builder.aiOnnxOpset11.convtranspose([d, f])
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    op_tester.setPatterns(['ConvTranspose'], enableRuntimeAsserts=False)
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern ConvFlipWeightsGradOp"))


def test_cos_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns([], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiOnnx.add([i1, w1], "test_add")
        o = builder.aiOnnx.cos([o], "test_cos")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern CosGradOp"))


def test_reciprocal_grad_error(op_tester):
    # Remove zeros for reciprocal
    d1 = np.random.rand(2, 7).astype(np.float32) + 1
    d2 = np.random.rand(2, 7).astype(np.float32) + 1

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns([], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiOnnx.add([i1, w1], "test_add")
        o = builder.aiOnnx.reciprocal([o], "test_reciprocal")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern ReciprocalGradOp"))


def test_sqrt_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns([], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiOnnx.add([i1, w1], "test_add")
        o = builder.aiOnnx.sqrt([o], "test_sqrt")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern SqrtGradOp"))


def test_subtract_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns([], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiOnnx.sub([i1, w1], "test_sub")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern SubtractArg1GradOp"))


def test_exp_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns([], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiOnnx.add([i1, w1], "test_add")
        o = builder.aiOnnx.exp([o], "test_exp")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern ExpGradOp"))


def test_expm1_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns([], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiOnnx.add([i1, w1], "test_add")
        o = builder.aiGraphcore.expm1([o], "test_expm1")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern Expm1GradOp"))


def test_cosh_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, op will be try to be created.
    op_tester.setPatterns([], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiOnnx.add([i1, w1], "test_add")
        o = builder.aiOnnx.cosh([o], "test_cosh")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern CoshOp"))


def test_log1p_grad_error(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d2 = np.random.rand(2, 7).astype(np.float32)

    # No pattern passes, grad op will not be created.
    op_tester.setPatterns([], enableRuntimeAsserts=False)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Add weight to force grad op creation
        w1 = builder.addInitializedInputTensor(d2)
        o = builder.aiOnnx.add([i1, w1], "test_add")
        o = builder.aiGraphcore.log1p([o], "test_log1p")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [None, None]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].endswith(
        "This op should have been removed by pattern Log1pGradOp"))
