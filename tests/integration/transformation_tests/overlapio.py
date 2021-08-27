# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
import platform
import pprint
import sys
from pathlib import Path

import numpy as np
import pytest

import popart

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


def get_model(size, batches_per_step, num_inputs, num_matmuls, tile_set,
              exchange_strategy):
    np.random.seed(num_inputs * num_matmuls)
    builder = popart.Builder()

    labels = builder.addInputTensor(
        popart.TensorInfo("INT32", [1, size]),
        popart.InputSettings(tile_set, exchange_strategy), "label")

    inputs = []
    weights = []
    s = []
    for i in range(num_inputs):
        x = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [1, size, size]),
            popart.InputSettings(tile_set, exchange_strategy), f"x{i}")
        inputs += [x]

        for j in range(num_matmuls):
            weight = np.random.normal(0, 0.05,
                                      (1, size, size)).astype(np.float32)
            w = builder.addInitializedInputTensor(weight, f"x{i}w{j}")
            weights += [w]

            x = builder.aiOnnx.matmul([w, x])
        s += [x]

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


def run_model(batches_per_step, accum_factor, replicas, tile_set,
              exchange_strategy):
    size = 64

    proto, inputs, weights, labels, dataFlow, loss, sum = get_model(
        size, batches_per_step, 4, 1, tile_set, exchange_strategy)

    opts = popart.SessionOptions()
    opts.enableExplicitMainLoops = True
    opts.useHostCopyOps = True
    opts.instrumentWithHardwareCycleCounter = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Auto

    # Both true & false should work - testing with false to avoid
    # host-cycle-overhead
    opts.rearrangeAnchorsOnHost = False
    opts.rearrangeStreamsOnHost = False

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
        deviceInfo=tu.create_test_device(numIpus=replicas,
                                         tilesPerIPU=tu.USE_ALL_TILES))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    np.random.seed(224488)

    session.weightsFromHost()

    warmup_iterations = 5
    calc_iterations = 20
    cycles = 0

    for i in range(warmup_iterations + calc_iterations):
        datainputs = {
            input: (np.random.normal(
                0, 0.05, (replicas * batches_per_step * accum_factor, 1, size,
                          size)).astype(np.float32))
            for input in inputs
        }
        datainputs[labels] = np.random.randint(
            0, size, (replicas * batches_per_step * accum_factor, 1, size))
        stepio = popart.PyStepIO(datainputs, anchors)
        session.run(stepio)
        if i >= warmup_iterations:
            cycles += session.getCycleCount()
    cycles = cycles / calc_iterations

    session.weightsToHost()
    weights_data = {
        w: np.zeros((1, size, size), dtype=np.float32)
        for w in weights
    }
    weights_read = popart.PyWeightsIO(weights_data)
    session.readWeights(weights_read)

    for w in weights_data:
        assert np.count_nonzero(np.isnan(weights_data[w])) == 0

    print("Cycles: ", cycles)

    return cycles, weights_data


def test_overlap_training():
    c0, w0 = run_model(4, 8, 1, popart.TileSet.Compute,
                       popart.ExchangeStrategy.JustInTime)
    c1, w1 = run_model(4, 8, 1, popart.TileSet.IO,
                       popart.ExchangeStrategy.JustInTime)
    c2, w2 = run_model(4, 8, 1, popart.TileSet.IO,
                       popart.ExchangeStrategy.OverlapInnerLoop)

    # Check all weights are equal
    for w in w0:
        assert np.allclose(w0[w], w1[w], equal_nan=False)
        assert np.allclose(w0[w], w2[w], equal_nan=False)

    # Check overlapped IO is at least 10% faster
    assert (c2 < 0.9 * c0)
    assert (c2 < 0.9 * c1)
