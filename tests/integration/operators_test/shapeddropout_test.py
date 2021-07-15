# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import operator
import popart
import pytest
import torch
import itertools
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


class ShapedDropoutHarness:
    def __init__(self,
                 input_data,
                 seed=8,
                 drop_ratio=0.5,
                 drop_shape=None,
                 batches_per_step=1,
                 replication_factor=1):
        self.input_data = input_data
        self.seed = seed
        self.drop_ratio = drop_ratio
        self.drop_shape = drop_shape
        self.batches_per_step = batches_per_step
        self.replication_factor = replication_factor

        if self.drop_shape is None:
            # PyTorch feature_dropout convention
            # Assumes that input data is NxM feature map
            drop_shape = np.array(input_data.shape)
            drop_shape[2:] = 1
            self.drop_shape = drop_shape

        self._setup()

    def _setup(self):
        ti = popart.TensorInfo("FLOAT", self.input_data.shape)
        builder = popart.Builder()
        self.input = builder.addInputTensor(ti)
        self.input_grad = popart.reservedGradientPrefix() + self.input

        self.output = builder.aiGraphcore.shapeddropout([self.input],
                                                        shape=self.drop_shape,
                                                        ratio=self.drop_ratio)

        self.loss = builder.aiGraphcore.identityloss([self.output])
        builder.addOutputTensor(self.loss)

        dfAnchors = [self.input, self.input_grad, self.output, self.loss]
        dfAnchors = {a: popart.AnchorReturnType("All") for a in dfAnchors}
        df = popart.DataFlow(self.batches_per_step, dfAnchors)

        model_proto = builder.getModelProto()
        all_patterns = popart.Patterns(popart.PatternsLevel.All)
        device = tu.create_test_device(self.replication_factor)

        session_opts = popart.SessionOptions()

        if self.replication_factor > 1:
            session_opts.enableReplicatedGraphs = True
            session_opts.replicatedGraphCount = self.replication_factor

        self.session = popart.TrainingSession(fnModel=model_proto,
                                              dataFlow=df,
                                              patterns=all_patterns,
                                              optimizer=popart.ConstSGD(0.1),
                                              loss=self.loss,
                                              userOptions=session_opts,
                                              deviceInfo=device)

        self.session.prepareDevice()
        self.session.setRandomSeed(self.seed)
        self.anchors = self.session.initAnchorArrays()
        batched_data = np.tile(
            self.input_data,
            [self.batches_per_step * self.replication_factor, 1, 1])
        self.stepio = popart.PyStepIO({self.input: batched_data}, self.anchors)

    def run(self):
        self.session.run(self.stepio)
        return self.anchors


def assert_not_equal(A, B):
    if np.array_equal(A, B):
        pytest.fail(("Arrays compare equal when expected not to:\n"
                     f"A = \n{A}\n\nB = \n{B}"))


# Check that shapeddropout is equal to an identity function in inference
@tu.requires_ipu_model
def test_shapeddropout_inference(op_tester):
    np.random.seed(10)
    d = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        ip = builder.addInputTensor(d)
        o = builder.aiGraphcore.shapeddropout([ip], shape=[1], ratio=0.5)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [d]

    # Without identity pattern...
    op_tester.run(init_builder, reference, 'infer')

    # ... and with identity pattern
    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


# Check that non-masked elements are scaled by 1/(1-ratio) just like in pytorch
@tu.requires_ipu
def test_shapeddropout_scaling(op_tester):
    # Use random input in the range [1., 2) so that comparisons with zero are valid
    np.random.seed(0)
    d = np.random.uniform(low=1.0, high=2.0, size=(8, 2)).astype(np.float32)
    prob = 0.5

    def init_builder(builder):
        ip = builder.addInputTensor(d)
        o = builder.aiGraphcore.shapeddropout([ip], shape=[8, 1], ratio=prob)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        # Use the IPU output to dropout the same elements on the original data
        r = d / (1. - prob)
        r[ref_data.getOutputTensor(0) == 0] = 0
        return [r]

    op_tester.run(init_builder, reference, seed=8, step_type='train')


# Check that entire rows are dropped in NxM matrix when the dropout shape is Nx1
@tu.requires_ipu
def test_shapeddropout_droprows():
    # Large number of rows for statistical convergence
    nrows = 1000
    data = np.ones([nrows, 3], dtype=np.float32)
    harness = ShapedDropoutHarness(data, seed=11583, drop_shape=[nrows, 1])
    anchors = harness.run()
    output = anchors[harness.output]

    # Count number of all zero rows to calculate the actual row-wise dropout ratio
    ndropped = np.sum(~output.any(axis=1))
    actual_ratio = ndropped / nrows
    np.testing.assert_allclose(actual_ratio, harness.drop_ratio, atol=0.02)


# Check that the same dropout mask is applied on forward and backward passes
@tu.requires_ipu
def test_shapeddropout_fwd_mask_eq_bwd():
    data = np.ones([4, 10], dtype=np.float32)
    harness = ShapedDropoutHarness(data, drop_shape=[1, 10])
    anchors = harness.run()
    input_grad = anchors[harness.input_grad]
    output = anchors[harness.output]

    np.testing.assert_equal(input_grad.nonzero(), output.nonzero())


# Check that a different mask is used every time session.run is called
@tu.requires_ipu
def test_shapeddropout_independent_rng_runs():
    np.random.seed(0)
    data = np.random.uniform(low=1.0, high=2.0, size=[100, 8])
    harness = ShapedDropoutHarness(data.astype(np.float32),
                                   drop_shape=[100, 1])
    anchors = harness.run()
    first_run = np.copy(anchors[harness.output])
    anchors = harness.run()
    second_run = np.copy(anchors[harness.output])
    assert_not_equal(first_run, second_run)


# Check that setting the rng seed produces repeatable shaped dropout masks
@tu.requires_ipu
def test_shapeddropout_set_seed():
    seed = 10
    data = np.ones([8, 10], dtype=np.float32)
    harness = ShapedDropoutHarness(data, drop_shape=[1, 10], seed=seed)
    anchors = harness.run()
    first_run = np.copy(anchors[harness.output])

    # Setting the seed and running again should produce the same random mask
    harness.session.setRandomSeed(seed)
    anchors = harness.run()
    second_run = np.copy(anchors[harness.output])
    np.testing.assert_equal(first_run, second_run)


# Check that the dropout mask is different for each batch when batches-per-step > 1
@tu.requires_ipu
def test_shapeddropout_independent_batches():
    np.random.seed(0)
    data = np.random.uniform(low=1.0, high=2.0, size=[1, 10, 2])
    bps = 2
    harness = ShapedDropoutHarness(data.astype(np.float32),
                                   batches_per_step=bps,
                                   drop_shape=data.shape)

    anchors = harness.run()
    out = anchors[harness.output]
    assert_not_equal(out[0], out[1])


# Check that having multiple shapeddropout ops with identical shape and ratio
# attributes will sample from independent PRNG streams.
@tu.requires_ipu
def test_shapeddropout_multiple(op_tester):
    np.random.seed(0)
    data = np.random.uniform(low=1.0, high=2.0, size=[30, 3])
    np_zero = np.array(0.0, dtype=np.float32)

    def init_builder(builder):
        # Model evaluates:
        #   sum( |sd1 - sd2| ) > 0
        T = builder.addInputTensor(data.astype(np.float32))
        sd1 = builder.aiGraphcore.shapeddropout([T], shape=[30, 1])
        sd2 = builder.aiGraphcore.shapeddropout([T], shape=[30, 1])
        diff = builder.aiOnnx.sub([sd1, sd2])
        loss = builder.aiOnnx.reducel1([diff], keepdims=False)
        zero = builder.addInputTensor(np_zero)
        out = builder.aiOnnx.greater([loss, zero])
        builder.addOutputTensor(out)
        return [loss, out]

    def reference(ref_data):
        return [None, np.array(True)]

    op_tester.run(init_builder, reference, step_type='train', seed=10)


# Check shapeddropout with replicated graphs:
#   - Each replica samples from an independent PRNG stream
#   - Multiple runs produce different shaped dropout masks
#   - Setting the seed produces the same shaped dropout masks
@tu.requires_ipu
def test_shapeddropout_replicated():
    session_seed = 8
    data = np.ones([4, 10], dtype=np.float32)
    harness = ShapedDropoutHarness(data,
                                   drop_shape=[1, 10],
                                   replication_factor=2,
                                   seed=session_seed)
    anchors = harness.run()
    output = np.copy(anchors[harness.output])
    assert_not_equal(output[0], output[1])

    # Another run should produce different results
    harness.run()
    second_run = np.copy(anchors[harness.output])
    assert_not_equal(output, second_run)

    # Resetting the random seed should reproduce the results from the first run
    harness.session.setRandomSeed(session_seed)
    harness.run()
    reset_run = anchors[harness.output]
    np.testing.assert_equal(output, reset_run)


# CPU test: check popart errors when ratio is outside the allowed range (0,1)
def test_shapeddropout_invalid_ratio():
    bad_ratio = 1.5
    with pytest.raises(popart.popart_exception) as e_info:
        ShapedDropoutHarness(np.zeros([2, 2]), drop_ratio=bad_ratio)

    exmsg = ("ratio value {} is not valid. "
             "Please use a value in the interval [0,1)").format(bad_ratio)
    assert e_info.value.args[0].endswith(exmsg)


# CPU test: check popart errors when ratio is outside the allowed range (0,1)
def test_shapeddropout_invalid_shape():
    bad_shape = [4, 4]
    with pytest.raises(popart.popart_exception) as e_info:
        ShapedDropoutHarness(np.zeros([2, 2]), drop_shape=bad_shape)

    exmsg = "ShapedDropout: incompatible input tensor and dropout shape."
    assert e_info.value.args[0].startswith(exmsg)
