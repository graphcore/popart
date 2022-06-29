# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path
import numpy as np
import pytest
import popart
import pva

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu
"""
Model:
        X0             X1    ... Xn  Labels
        |              |         |    |
X0W0 - MatMul  X1W0 - MatMul ... .    |
        |              |         |    |
X0W1 - MatMul  X1W1 - Matmul ... .    |
 .      |              |         |    |
 .      .              .     ... .    |
 .      |              |         |    |
X0Wm - MatMul  X1Wm - MatMul ... .    |
        |              |         |    |
        |              |         |    |
        '------------ Sum ---...-'    |
                     / |              |
                  /    |              |
            (Anchor) Softmax          |
                       |              |
                    NLLLoss ----------'
                       |
                    (Anchor)

Expectation: Anchor streams (Anchor) and inputs (X0, ..., X1, Labels),
             which are copied from the host to the IPU (streams),
             or from the IPU to the host (anchors),
             will overlap in time with the MatMul operations.
"""


def get_compute_io_overlap_percentage(report, runIndex):
    """
    Returns the percentage of compute and IO overlapped in the execution
    report.
    """

    # Execution steps for the run at runIndex
    steps = report.execution.runs[runIndex].steps
    computeIntervals = []
    streamCopyIntervals = []

    class IntervalVisitor(pva.ProgramVisitor):
        streamCopyStart = 0
        streamCopyMid = False

        def __init__(self, cyclesFrom, cyclesTo):
            self.cyclesFrom = cyclesFrom
            self.cyclesTo = cyclesTo
            super(IntervalVisitor, self).__init__()

        def visitOnTileExecute(self, _):
            computeIntervals.append([self.cyclesFrom.max, self.cyclesTo.max])

        def visitStreamCopyMid(self, _):
            streamCopyIntervals.append(
                [self.cyclesFrom.max, self.cyclesTo.max])

    for step in steps:
        ipu = step.ipus[0]
        f = ipu.activeCycles.cyclesFrom
        t = ipu.activeCycles.cyclesTo
        v = IntervalVisitor(f, t)
        step.program.accept(v)

        def checkOverlap(low1, high1, low2, high2):
            return low1 < high2 and low2 < high1

        def getOverlap(low1, high1, low2, high2):
            overlap = min(high1, high2) - max(low1, low2)
            return overlap

        # Compute amount of overlap of compute and stream copy (mid) intervals
        overlap = 0
        for compute in computeIntervals:
            for stream in streamCopyIntervals:
                if checkOverlap(compute[0], compute[1], stream[0], stream[1]):
                    overlap += getOverlap(compute[0], compute[1], stream[0],
                                          stream[1])

    computeTotal, streamCopyTotal = (
        sum(i[1] - i[0] for i in intervals)
        for intervals in [computeIntervals, streamCopyIntervals])

    # Return percentage overlap
    return max(overlap / streamCopyTotal, overlap / computeTotal)


def get_model(size, batches_per_step, num_inputs, num_matmuls, tile_set,
              exchange_strategy, pipelining):
    np.random.seed(num_inputs * num_matmuls)
    builder = popart.Builder()

    inputs = []
    weights = []

    labels = builder.addInputTensor(
        popart.TensorInfo("INT32", [1, size]),
        popart.InputSettings(tile_set, exchange_strategy), "label")
    s = []
    for i in range(num_inputs):
        x = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [1, size, size]),
            popart.InputSettings(tile_set, exchange_strategy), f"x{i}")
        inputs += [x]

        for j in range(num_matmuls):
            with builder.virtualGraph(
                    j if pipelining else 0), builder.pipelineStage(j):
                weight = np.random.normal(0, 0.05,
                                          (1, size, size)).astype(np.float32)
                w = builder.addInitializedInputTensor(weight, f"x{i}w{j}")
                weights += [w]

                x = builder.aiOnnx.matmul([w, x])
        s += [x]

    with builder.virtualGraph(num_matmuls -
                              1 if pipelining else 0), builder.pipelineStage(
                                  num_matmuls - 1):
        sum = builder.aiOnnx.sum(s)
        probs = builder.aiOnnx.softmax([sum])
        loss = builder.aiGraphcore.nllloss([probs, labels])

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(
        batches_per_step, {
            loss: popart.AnchorReturnType("All", tile_set, exchange_strategy),
            sum: popart.AnchorReturnType("All", tile_set, exchange_strategy),
        })

    return proto, inputs, weights, labels, dataFlow, loss, sum


def run_model(tmpdir, batches_per_step, accum_factor, replicas, tile_set,
              exchange_strategy, pipelining):
    size = 64
    num_inputs = 4
    num_matmuls = 2 if pipelining else 1

    proto, inputs, weights, labels, dataFlow, loss, _ = get_model(
        size, batches_per_step, num_inputs, num_matmuls, tile_set,
        exchange_strategy, pipelining)

    opts = popart.SessionOptions()

    opts.enableExplicitIR(True)
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    if pipelining:
        opts.enablePipelining = pipelining
        opts.autoRecomputation = popart.RecomputationType.Pipeline
        ipus_per_replica = num_matmuls
    else:
        ipus_per_replica = 1

    opts.instrumentWithHardwareCycleCounter = False

    # Should work with both rearrangeOnHost = `True`/`False`.
    # Testing with `False` because it is more sensitive to tensor layouts and
    # therefore brittle.
    # If `False` works, `True` should work too
    opts.rearrangeAnchorsOnHost = False
    opts.rearrangeStreamsOnHost = False

    # Set session options to generate the report
    tu.set_autoreport_options(opts, tmpdir, output_execution_profile=True)

    if accum_factor > 1:
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = accum_factor

    if tile_set == popart.TileSet.IO:
        opts.numIOTiles = 128
    else:
        opts.numIOTiles = 0

    if replicas > 1:
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = replicas

    pat = popart.Patterns(popart.PatternsLevel.Default)

    with tu.create_test_device(numIpus=replicas * ipus_per_replica,
                               tilesPerIPU=tu.USE_ALL_TILES) as device:
        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=dataFlow,
            userOptions=opts,
            loss=loss,
            optimizer=popart.ConstSGD(1e-6),
            patterns=pat,
            # Trying to use less than all the tiles throw an error like
            #   popart_core.poplar_exception: Trying to access tile 72 on IPU
            #   0 but the virtual graph only covers the following tiles on
            #   that IPU: 0-63
            # The error happens in a call to poplar made by gcl::perIPUTiles.
            deviceInfo=device)

        anchors = session.initAnchorArrays()

        session.prepareDevice()

        np.random.seed(224488)

        session.weightsFromHost()

        warmup_iterations = 1
        calc_iterations = 1

        for _ in range(warmup_iterations + calc_iterations):
            datainputs = {
                input: (np.random.normal(
                    0, 0.05, (replicas * batches_per_step * accum_factor, 1,
                              size, size)).astype(np.float32))
                for input in inputs
            }
            datainputs[labels] = np.random.randint(
                0, size, (replicas * batches_per_step * accum_factor, 1, size))
            stepio = popart.PyStepIO(datainputs, anchors)
            session.run(stepio)

        session.weightsToHost()
        weights_data = {
            w: np.zeros((1, size, size), dtype=np.float32)
            for w in weights
        }
        weights_read = popart.PyWeightsIO(weights_data)
        session.readWeights(weights_read)

        for w in weights_data:
            assert np.count_nonzero(np.isnan(weights_data[w])) == 0

    report = session.getReport()

    overlapPercentage = get_compute_io_overlap_percentage(
        report, warmup_iterations)

    return overlapPercentage, weights_data


@pytest.mark.parametrize("pipelining", [False, True])
def test_overlap_training(tmpdir, pipelining):
    print("Temporary directory:", tmpdir)

    batches_per_step = 1
    accum_factor = 16
    replicas = 2

    p0, w0 = run_model(tmpdir, batches_per_step, accum_factor, replicas,
                       popart.TileSet.Compute,
                       popart.ExchangeStrategy.JustInTime, pipelining)
    p1, w1 = run_model(tmpdir, batches_per_step, accum_factor, replicas,
                       popart.TileSet.IO, popart.ExchangeStrategy.JustInTime,
                       pipelining)
    p2, w2 = run_model(tmpdir, batches_per_step, accum_factor, replicas,
                       popart.TileSet.IO,
                       popart.ExchangeStrategy.OverlapInnerLoop, pipelining)

    # Reference values (MK2 C200):
    # p0 0.0
    # p1 0.14851003840357785
    # p2 0.4811525129982669

    print("p0", p0)
    print("p1", p1)
    print("p2", p2)

    # Check all weights are equal
    for w in w0:
        assert np.allclose(w0[w], w1[w], equal_nan=False)
        assert np.allclose(w0[w], w2[w], equal_nan=False)

    # Check that overlapped IO increases compute-IO overlap percentage
    # significantly
    # At least 25% more than not using IO tiles at all
    assert p2 - p0 > 0.25
    # At least 25% more than using IO tiles but no overlap strategy
    assert p2 - p1 > 0.25
