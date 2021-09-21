# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import collections
import itertools
import numpy as np
import pytest
import popart
import onnx
"""
NOTE: In theory random ops can produce identical output despite having different
seeds, so checking for distinct outputs as a way of testing probabilistic
behaviour is not always a good idea. We mostly use dropouts with 100 elements
because the probability of two outputs being the same for different seeds is
~0.5^100. We have fixed seeds in this file to ensure tests are deterministic
and there is no small probability of failure.
"""

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

# Defines a slice of a tensor that is divided into "total_slices" equal parts
# in the first axis by the 0-offset value "index".
Slice = collections.namedtuple("Slice", ["index", "total_slices"])

# An object to hold the result of the "run_model" function.
Run = collections.namedtuple("Run",
                             ["anchors", "seed", "steps", "random_outs"])

# Constant value for taking all of a tensor (first index, 1 slice in total).
whole = Slice(index=0, total_slices=1)


@pytest.mark.parametrize("useHostCopyOps", [False, True])
def test_random_behaviour_governed_by_seed(useHostCopyOps):
    """
    Create a model with 1 dropout and run it a few times with different session
    seeds. Check that using the same seed results in the same output and using
    different seeds results in different outputs.
    """

    options = popart.SessionOptions()
    options.useHostCopyOps = useHostCopyOps

    num_steps = 1
    num_elems = 100

    d0 = np.asarray([1.0] * num_elems * num_steps).astype(np.float32)

    def model_fn(builder):
        main_in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))
        main_dropout0 = builder.aiOnnx.dropout([main_in0], 1, 0.5)[0]
        loss = builder.aiGraphcore.identityloss(
            [main_dropout0], reduction=popart.ReductionType.Mean)
        return loss, {main_in0: d0}, {'main_dropout0': main_dropout0}

    run0 = run_model(builder_fn=model_fn,
                     steps=num_steps,
                     seed=0,
                     options=options)
    run1 = run_model(builder_fn=model_fn,
                     steps=num_steps,
                     seed=0,
                     options=options)
    run2 = run_model(builder_fn=model_fn,
                     steps=num_steps,
                     seed=11583,
                     options=options)

    # run0 and run1 (which use the same seed) should agree on each step.
    for s in range(num_steps):
        # yapf: disable
        ensure_mask_equal(run0, 'main_dropout0', s, whole,
                          run1, 'main_dropout0', s, whole)
        # yapf: enable

    # run0 and run2 (which use different seeds) should disagree on every step.
    for s0, s1 in itertools.product(range(num_steps), repeat=2):
        # yapf: disable
        ensure_mask_not_equal(run0, 'main_dropout0', s0, whole,
                              run2, 'main_dropout0', s1, whole)
        # yapf: enable


def test_distinct_random_behaviour_per_step():
    """
    Create a model with 1 dropout. Check that every step of a run results in a
    different random mask. Also check that the fwd dropout mask always matches
    the backward one.
    """

    num_steps = 2
    num_elems = 100

    d0 = np.asarray([1.0] * num_elems * num_steps).astype(np.float32)

    def model_fn(builder):
        main_in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))
        main_dropout0 = builder.aiOnnx.dropout([main_in0], 1, 0.5)[0]
        loss = builder.aiGraphcore.identityloss(
            [main_dropout0], reduction=popart.ReductionType.Mean)
        return loss, {
            main_in0: d0
        }, {
            'main_dropout0': main_dropout0,
            'main_dropout0_grad': popart.reservedGradientPrefix() + main_in0
        }

    run0 = run_model(builder_fn=model_fn, steps=num_steps, seed=0)

    # forwards/backwards mask should be the same
    for s in range(num_steps):
        # yapf: disable
        ensure_mask_equal(run0, 'main_dropout0', s, whole,
                          run0, 'main_dropout0_grad', s, whole)
        # yapf: enable

    # no steps should be the same
    for s0, s1 in itertools.product(range(num_steps), repeat=2):
        if s0 < s1:
            # yapf: disable
            ensure_mask_not_equal(run0, 'main_dropout0', s0, whole,
                                  run0, 'main_dropout0', s1, whole)
            # yapf: enable


def test_distinct_random_behaviour_per_op():
    """
    Create a model with 2 dropouts and check their random behaviour is different
    to each other and between them, no random behaviour is repeated. Also check
    that the masks used in the forward and backwards pass match for each op.
    """

    num_steps = 1
    num_elems = 100

    d0 = np.asarray([1.0] * num_elems * num_steps).astype(np.float32)

    def model_fn(builder):
        # Model with 1 dropout.
        main_in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))
        main_ident0 = builder.aiGraphcore.scale([main_in0], 0.1)
        main_ident1 = builder.aiGraphcore.scale([main_in0], 0.2)
        main_dropout0 = builder.aiOnnx.dropout([main_ident0], 1, 0.5)[0]
        main_dropout1 = builder.aiOnnx.dropout([main_ident1], 1, 0.5)[0]
        main_add = builder.aiOnnx.add([main_dropout0, main_dropout1])
        loss = builder.aiGraphcore.identityloss(
            [main_add], reduction=popart.ReductionType.Mean)
        return loss, {
            main_in0: d0
        }, {
            'main_dropout0': main_dropout0,
            'main_dropout1': main_dropout1,
            'main_dropout0_grad':
            popart.reservedGradientPrefix() + main_ident0,
            'main_dropout1_grad': popart.reservedGradientPrefix() + main_ident1
        }

    run0 = run_model(builder_fn=model_fn, steps=num_steps, seed=0)

    # ops should have different seeds.
    for s0, s1 in itertools.product(range(num_steps), repeat=2):
        # yapf: disable
        ensure_mask_not_equal(run0, 'main_dropout0', s0, whole,
                              run0, 'main_dropout1', s1, whole)
        # yapf: enable

    # but fwd/bwd mask should match for each op.
    for s in range(num_steps):
        # yapf: disable
        ensure_mask_equal(run0, 'main_dropout0', s, whole,
                          run0, 'main_dropout0_grad', s, whole)
        ensure_mask_equal(run0, 'main_dropout1', s, whole,
                          run0, 'main_dropout1_grad', s, whole)
        # yapf: enable


def test_random_op_in_subgraph():
    """
    Create a model with 1 dropout in a subgraph and make sure that each step has
    distinct random behaviour and fwd/bwd mask matches.
    """

    num_steps = 2
    num_elems = 100

    d0 = np.asarray([1.0] * num_elems * num_steps).astype(np.float32)

    def model_fn(builder):
        # Model with 1 dropout.
        main_in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))

        sg0_builder = builder.createSubgraphBuilder()
        sg0_in0 = sg0_builder.addUntypedInputTensor()
        sg0_rnd0 = sg0_builder.aiOnnx.dropout([sg0_in0], 1, 0.5)[0]
        sg0_builder.addOutputTensor(sg0_rnd0)
        main_dropout0 = builder.aiGraphcore.call([main_in0], 1, sg0_builder)[0]
        loss = builder.aiGraphcore.identityloss(
            [main_dropout0], reduction=popart.ReductionType.Mean)
        return loss, {
            main_in0: d0
        }, {
            'main_dropout0': main_dropout0,
            'main_dropout0_grad': popart.reservedGradientPrefix() + main_in0
        }

    run0 = run_model(builder_fn=model_fn, steps=num_steps, seed=0)

    # no step should yield the same mask.
    for s0, s1 in itertools.product(range(num_steps), repeat=2):
        if s0 < s1:
            # yapf: disable
            ensure_mask_not_equal(run0, 'main_dropout0', s0, whole,
                                  run0, 'main_dropout0', s1, whole)
            # yapf: enable

    # op and grad op should have the same mask.
    for s in range(num_steps):
        # yapf: disable
        ensure_mask_equal(run0, 'main_dropout0', s, whole,
                          run0, 'main_dropout0_grad', s, whole)
        # yapf: enable


def test_two_calls_to_subgraph_with_random_op():
    """
    Create a model with 1 dropout in a subgraph, call the subgraph twice and
    make sure both calls have distinct random behaviour and the fwd/bwd mask
    always matches for both calls.
    """

    num_steps = 1
    num_elems = 100

    d0 = np.asarray([1.0] * num_elems * num_steps).astype(np.float32)

    def model_fn(builder):
        # Model with 1 dropout.
        main_in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))

        sg0_builder = builder.createSubgraphBuilder()
        sg0_in0 = sg0_builder.addUntypedInputTensor()
        sg0_rnd0 = sg0_builder.aiOnnx.dropout([sg0_in0], 1, 0.5)[0]
        sg0_builder.addOutputTensor(sg0_rnd0)
        main_ident0 = builder.aiGraphcore.scale([main_in0], 0.1)
        main_ident1 = builder.aiGraphcore.scale([main_in0], 0.2)
        main_dropout0 = builder.aiGraphcore.call([main_ident0], 1,
                                                 sg0_builder)[0]
        main_dropout1 = builder.aiGraphcore.call([main_ident1], 1,
                                                 sg0_builder)[0]
        main_add = builder.aiOnnx.add([main_dropout0, main_dropout1])
        loss = builder.aiGraphcore.identityloss(
            [main_add], reduction=popart.ReductionType.Mean)
        return loss, {
            main_in0: d0
        }, {
            'main_dropout0': main_dropout0,
            'main_dropout1': main_dropout1,
            'main_dropout0_grad':
            popart.reservedGradientPrefix() + main_ident0,
            'main_dropout1_grad': popart.reservedGradientPrefix() + main_ident1
        }

    run0 = run_model(builder_fn=model_fn, steps=num_steps, seed=0)

    # calls should not agree on seeds for any steps.
    for s0, s1 in itertools.product(range(num_steps), repeat=2):
        # yapf: disable
        ensure_mask_not_equal(run0, 'main_dropout0', s0, whole,
                              run0, 'main_dropout1', s1, whole)
        # yapf: enable

    # but fwd/bwd mask should match for each individual op.
    for s in range(num_steps):
        # yapf: disable
        ensure_mask_equal(run0, 'main_dropout0', s, whole,
                          run0, 'main_dropout0_grad', s, whole)
        ensure_mask_equal(run0, 'main_dropout1', s, whole,
                          run0, 'main_dropout1_grad', s, whole)
        # yapf: enable


def test_nesting_of_subgraphs():
    """
    Create a model with 1 dropout that is called in a subgraph which
    subsequently is called within another subgraph. The middle subgraph has no
    random ops.
    """

    num_steps = 2
    num_elems = 100

    d0 = np.asarray([1.0] * num_elems * num_steps).astype(np.float32)

    def model_fn(builder):
        # Model with 1 dropout.
        main_in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))

        sg0_builder = builder.createSubgraphBuilder()
        sg0_in0 = sg0_builder.addUntypedInputTensor()
        sg0_rnd0 = sg0_builder.aiOnnx.dropout([sg0_in0], 1, 0.5)[0]
        sg0_builder.addOutputTensor(sg0_rnd0)
        sg1_builder = builder.createSubgraphBuilder()
        sg1_in0 = sg1_builder.addUntypedInputTensor()
        sg1_call0 = sg1_builder.aiGraphcore.call([sg1_in0], 1, sg0_builder)[0]
        sg1_builder.addOutputTensor(sg1_call0)
        main_dropout0 = builder.aiGraphcore.call([main_in0], 1, sg1_builder)[0]
        loss = builder.aiGraphcore.identityloss(
            [main_dropout0], reduction=popart.ReductionType.Mean)
        return loss, {
            main_in0: d0
        }, {
            'main_dropout0': main_dropout0,
            'main_dropout0_grad': popart.reservedGradientPrefix() + main_in0
        }

    run0 = run_model(builder_fn=model_fn, steps=num_steps, seed=0)

    # no step should yield the same mask.
    for s0, s1 in itertools.product(range(num_steps), repeat=2):
        if s0 < s1:
            # yapf: disable
            ensure_mask_not_equal(run0, 'main_dropout0', s0, whole,
                                  run0, 'main_dropout0', s1, whole)
            # yapf: enable

    # op and grad op should have the same mask.
    for s in range(num_steps):
        # yapf: disable
        ensure_mask_equal(run0, 'main_dropout0', s, whole,
                          run0, 'main_dropout0_grad', s, whole)
        # yapf: enable


def test_distinct_random_behaviour_with_subgraphs():
    """
    Test to check that we're not just using a simple 'adding' scheme to
    modify seeds. That is, if we have model like this:

    def main():
        sg1() <- modifier constant 0
        sg1() <- modifier constant 1

    def sg1():
        dropout() <- modifier constant 0
        dropout() <- modifier constant 1

    Then if ModifyRandomSeedOp simply adds the constant to the seed (this is
    something we previously did) we end up the second dropout in the first
    sg1() call using the same seed as the first dropout in the second sg1()
    call.

    We don't check grads masks here because we checked that in enough other
    tests.
    """

    num_steps = 1
    num_elems = 100

    d0 = np.asarray([1.0] * num_elems * num_steps).astype(np.float32)

    def model_fn(builder):
        # Model with 1 dropout.
        main_in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))

        sg0_builder = builder.createSubgraphBuilder()
        sg0_in0 = sg0_builder.addUntypedInputTensor()
        sg0_rnd0 = sg0_builder.aiOnnx.dropout([sg0_in0], 1, 0.5)[0]
        sg0_rnd1 = sg0_builder.aiOnnx.dropout([sg0_in0], 1, 0.5)[0]
        sg0_builder.addOutputTensor(sg0_rnd0)
        sg0_builder.addOutputTensor(sg0_rnd1)
        call_out = builder.aiGraphcore.call([main_in0], 2, sg0_builder)
        main_dropout0, main_dropout1 = call_out
        call_out = builder.aiGraphcore.call([main_in0], 2, sg0_builder)
        main_dropout2, main_dropout3 = call_out
        main_add0 = builder.aiOnnx.add([main_dropout0, main_dropout1])
        main_add1 = builder.aiOnnx.add([main_dropout2, main_dropout3])
        main_add1 = builder.aiOnnx.add([main_add0, main_add1])
        loss = builder.aiGraphcore.identityloss(
            [main_add1], reduction=popart.ReductionType.Mean)
        return loss, {
            main_in0: d0
        }, {
            'main_dropout0': main_dropout0,
            'main_dropout1': main_dropout1,
            'main_dropout2': main_dropout2,
            'main_dropout3': main_dropout3,
            # Grad not checked but needed here because of T36121
            'main_dropout0_grad': popart.reservedGradientPrefix() + main_in0
        }

    run0 = run_model(builder_fn=model_fn, steps=num_steps, seed=0)

    # ops should have different seeds.
    for s0, s1 in itertools.product(range(num_steps), repeat=2):
        # yapf: disable
        for d0, d1 in itertools.product(range(4), repeat=2):
            if d0 < d1:
                ensure_mask_not_equal(run0, f'main_dropout{d0}', s0, whole,
                                      run0, f'main_dropout{d1}', s1, whole)
        # yapf: enable


def test_random_op_in_loop_body():
    """
    Create a model with 1 uniform random op in a loop body and check each
    loop iteration yields distinct random behaviour.

    NOTE: This can't currently use a training session because LoopOp does not
    yet work with autodiff. Because of this we can't use Dropout here
    because Dropouts don't have any effect in inference mode.
    """

    num_steps = 2
    num_elems = 100
    num_iters = 10

    d0 = np.asarray([1.0] * num_elems * num_steps).astype(np.float32)

    def model_fn(builder):
        # Model with 1 dropout.
        main_in0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))

        # Num loop iterations.
        M = builder.aiOnnx.constant(np.array(num_iters).astype(np.int64), "M")
        cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond")

        # loop body subgraph
        loop_builder = builder.createSubgraphBuilder()
        loop_builder.setGraphName("loop_body")
        # Trip counter.
        loop_iters = loop_builder.addInputTensor(popart.TensorInfo(
            "INT64", []))
        # Termination condition.
        loop_cond = loop_builder.addInputTensor(popart.TensorInfo("BOOL", []))
        # explicit input, loop carried.
        loop_res = loop_builder.addInputTensor(
            popart.TensorInfo("FLOAT", [num_elems]))
        # do the dropout (note: use tensor from parent scope as input).
        loop_rnd0 = loop_builder.aiOnnx.randomuniform(shape=[num_elems])
        # Calculate dynamic slice offset.
        offset = loop_builder.aiGraphcore.scale([loop_iters], num_elems)
        # Dynamic slice to stack the dropout result tensors.
        loop_res = loop_builder.aiGraphcore.dynamicupdate(
            [loop_res, offset, loop_rnd0],
            axes=[0],
            sizes=[num_elems],
            noOverlap=True)
        loop_builder.addOutputTensor(loop_cond)
        loop_builder.addOutputTensor(loop_res)

        loop_var = builder.aiGraphcore.init([num_iters * num_elems],
                                            popart.DataType.FLOAT,
                                            popart.InitType.NoInit)

        main_rnd = builder.aiOnnx.loop([M, cond, loop_var], 1, loop_builder)[0]
        loss = builder.aiGraphcore.identityloss(
            [main_rnd], reduction=popart.ReductionType.Mean)
        return loss, {main_in0: d0}, {'loss': loss, 'main_rnd': main_rnd}

    run0 = run_model(builder_fn=model_fn,
                     steps=num_steps,
                     seed=0,
                     training=False)

    # no step should yield the same mask.
    for s0, s1 in itertools.product(range(num_steps), repeat=2):
        for i0, i1 in itertools.product(range(num_iters), repeat=2):
            if s0 <= s1 and i0 <= i1 and (s0 != s1 or i0 != i1):
                # yapf: disable
                slice0 = Slice(index=i0, total_slices=num_iters)
                slice1 = Slice(index=i1, total_slices=num_iters)
                ensure_value_not_equal(run0, 'main_rnd', s0, slice0,
                                       run0, 'main_rnd', s1, slice1)
                # yapf: enable


def run_model(builder_fn,
              steps,
              seed,
              training=True,
              options=popart.SessionOptions()):
    """
    Helper function that runs a model and returns the anchors.

      builder_fn - a function that takes a PopART builder and returns a tuple
                   comprising a loss, a dictionary of inputs and a dictionary
                   that maps python variable names to PopART tensor IDs for
                   anchors.
      steps      - number of batches per step
      seed       - random seed to pass to the PopART session.

    Returns a named tuple with .anchors being the anchors and .seed being the
    seed used.
    """

    builder = popart.Builder()
    loss, inputs, random_outs = builder_fn(builder)
    dataFlow = popart.DataFlow(
        steps,
        {op[1]: popart.AnchorReturnType("ALL")
         for op in random_outs.items()})

    proto = builder.getModelProto()
    optimizer = popart.SGD({"defaultLearningRate": (0.1, True)})
    patterns = popart.Patterns()

    device = tu.create_test_device(1, pattern=popart.SyncPattern.Full)

    if training:
        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=dataFlow,
                                         userOptions=options,
                                         loss=loss,
                                         optimizer=optimizer,
                                         patterns=patterns,
                                         deviceInfo=device)
    else:
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=options,
                                          patterns=patterns,
                                          deviceInfo=device)

    session.prepareDevice()
    session.weightsFromHost()
    session.setRandomSeed(seed)
    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO(inputs, anchors)
    session.run(stepio)

    return Run(anchors=anchors,
               seed=seed,
               steps=steps,
               random_outs=random_outs)


def get_anchor_step(run, anchor, step, slice):
    """
    Helper function to get a specific tensor's batch result from a Run object.

    run    - the Run object to take the tensor from.
    anchor - the tensor name of the anchors to return values for.
    step   - the batch number to return values for.
    slice  - gives you an ability to slice the value in the first axis if a
             tensor is, say, a number of stacked dropout results from a loop.
             The slice is defined by a tuple comprising
             (<index of slice>/<number of slices>) and assumes it always slices
             in the first axis and in equal parts.
    """
    out = run.anchors[run.random_outs[anchor]]
    if run.steps > 1:
        out = out[step]

    # Take a slice of the tensor.
    slice_size = out.shape[0] // slice.total_slices
    out = out[slice.index * slice_size:(slice.index + 1) * slice_size]
    return out


def slice_to_str(slice):
    """
    A way of printing slices relatively unverbosely. Most of the time, we don't
    use the slices feature in tests, and it's clearer to print "whole" as 
    opposed to "Slice(index=0, total_slices=1)".
    """
    if slice == whole:
        return 'whole'
    else:
        return f"({slice.index}/{slice.total_slices})"


def ensure_mask_equal(run0, anchor0, step0, slice0, run1, anchor1, step1,
                      slice1):
    """
    Helper function to compare two random op outputs and assert they are the same.

    See get_anchor_step for details on arguments.
    """
    tensor0 = get_anchor_step(run0, anchor0, step0, slice0) == 0
    tensor1 = get_anchor_step(run1, anchor1, step1, slice1) == 0
    assert (np.array_equal(tensor0, tensor1)), f"""
      Expected output '{anchor0}', step {step0}, slice={slice_to_str(slice0)} (seed={run0.seed})
        {tensor0}
      to match output '{anchor1}', step {step1}, slice={slice_to_str(slice1)}  (seed={run1.seed})
        {tensor1}"""


def ensure_mask_not_equal(run0, anchor0, step0, slice0, run1, anchor1, step1,
                          slice1):
    """
    Helper function to compare two random op outputs, ensuring they are different.

    See get_anchor_step for details on arguments.
    """
    tensor0 = get_anchor_step(run0, anchor0, step0, slice0) == 0
    tensor1 = get_anchor_step(run1, anchor1, step1, slice1) == 0
    assert (not np.array_equal(tensor0, tensor1)), f"""
      Expected output '{anchor0}', step {step0}, slice={slice_to_str(slice0)} (seed={run0.seed})
        {tensor0}
      to be different from output '{anchor1}', step {step1}, slice={slice_to_str(slice1)} (seed={run1.seed})
        {tensor1}"""


def ensure_value_not_equal(run0, anchor0, step0, slice0, run1, anchor1, step1,
                           slice1):
    """
    Helper function to compare two random op outputs, ensuring they are different.

    See get_anchor_step for details on arguments.
    """
    tensor0 = get_anchor_step(run0, anchor0, step0, slice0)
    tensor1 = get_anchor_step(run1, anchor1, step1, slice1)
    assert (not np.array_equal(tensor0, tensor1)), f"""
      Expected output '{anchor0}', step {step0}, slice={slice_to_str(slice0)} (seed={run0.seed})
        {tensor0}
      to be different from output '{anchor1}', step {step1}, slice={slice_to_str(slice1)} (seed={run1.seed})
        {tensor1}"""
