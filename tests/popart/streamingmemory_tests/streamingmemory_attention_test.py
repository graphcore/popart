# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import popart
import pprint
import json
import platform
import itertools
import torch
import onnx
from onnx import numpy_helper

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
from _ast import Or
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# Attention Op from BERT
def attention_onnx(builder, qkv, mask, batch_size, sequence_length,
                   hidden_size, attention_heads, qkv_length):
    comb_shape = [batch_size, sequence_length, attention_heads, qkv_length]

    def extract_heads(tensor, index, hidden_size, transpose=False):
        tensor = builder.aiOnnxOpset9.slice([qkv],
                                            axes=[1],
                                            starts=[index * hidden_size],
                                            ends=[(index + 1) * hidden_size])
        tensor = builder.reshape_const(builder.aiOnnx, [tensor], comb_shape)
        perm = [0, 2, 1, 3] if not transpose else [0, 2, 3, 1]
        return builder.aiOnnx.transpose([tensor], perm=perm)

    q, kt, v = [extract_heads(qkv, i, hidden_size, i == 1) for i in range(3)]

    # Attention calculation
    with builder.nameScope("Z"):
        x = builder.aiOnnx.matmul([q, kt])

        c = builder.aiOnnx.constant(
            np.array(1 / np.sqrt(qkv_length)).astype(np.float32), "C")
        x = builder.aiOnnx.mul([x, c])

        x = builder.aiOnnx.add([x, mask], "ApplyMask")

        x = builder.aiOnnx.softmax([x], axis=-1)

        # x[batch_size, attention_heads, sequence_length, sequence_length] * v[batch_size, attention_heads, sequence_length, qkv_length]
        z = builder.aiOnnx.matmul([x, v])

        # [batch_size, attention_heads, sequence_length, qkv_length] -> [batch_size, sequence_length, attention_heads, qkv_length]
        z = builder.aiOnnx.transpose([z], perm=[0, 2, 1, 3])
        # [batch_size, sequence_length, attention_heads, qkv_length] -> [batch_size*sequence_length, attention_heads*qkv_length]
        z = builder.reshape_const(builder.aiOnnx, [z],
                                  [sequence_length * batch_size, hidden_size])
    return z


@tu.requires_ipu
def test_attention_streamingmemory(tmpdir):
    np.random.seed(0XDEAD1337)
    batches_per_step = 5
    batch_size = 8
    hidden_size = 16
    sequence_length = 8
    attention_heads = 4
    qkv_length = hidden_size / attention_heads

    input_shape = [batch_size * sequence_length, hidden_size]
    mask_shape = [batch_size, 1, 1, sequence_length]

    qkv_data = np.random.normal(
        0, 0.02, [hidden_size, hidden_size * 3]).astype(np.float32)

    r = np.arange(0, sequence_length)
    r = np.reshape(batch_size * [r], mask_shape)
    masks = []
    for i in range(batches_per_step):
        masks.append(np.less(r, i).astype(np.float32))
    mask_data = (1 - np.stack(masks)) * -1000.0

    input_data = np.random.normal(
        0, 0.02, [batches_per_step] + input_shape).astype(np.float32)

    def run_test(index, options):
        per_replica_batch_size = batch_size / options["replication"]
        model_input_shape = input_shape[:]
        model_input_shape[0] = int(
            model_input_shape[0] / options["replication"])
        model_mask_shape = mask_shape[:]
        model_mask_shape[0] = int(model_mask_shape[0] / options["replication"])

        stride = 2 // options["stages"]
        if "stride" in options and options["stride"]:
            stride = options["stride"]

        builder = popart.Builder(opsets={
            "ai.onnx": 9,
            "ai.onnx.ml": 1,
            "ai.graphcore": 1
        })

        mask = builder.addInputTensor(
            popart.TensorInfo("FLOAT", model_mask_shape), "mask")
        x_in = builder.addInputTensor(
            popart.TensorInfo("FLOAT", model_input_shape), "x_in")

        anchors = {}
        x = x_in
        for i in range(options["numLayers"]):
            qkv = builder.addInitializedInputTensor(qkv_data, f"qkv_{i}")
            anchors[popart.reservedGradientPrefix() +
                    qkv] = popart.AnchorReturnType("All")

            vgid = (i % options["stages"]) if options["phasedExecution"] else i

            with builder.virtualGraph(vgid), builder.executionPhase(
                    i * stride):
                x = builder.aiOnnx.matmul([x, qkv])
                x = attention_onnx(builder, x, mask, per_replica_batch_size,
                                   sequence_length, hidden_size,
                                   attention_heads, qkv_length)

        vgid = ((options["numLayers"] - 1) % options["stages"]
                ) if options["phasedExecution"] else options["numLayers"] - 1

        with builder.virtualGraph(vgid), builder.executionPhase(
            (options["numLayers"] - 1) * stride):
            l1 = builder.aiGraphcore.l1loss([x], 0.2, popart.ReductionType.Sum)

        proto = builder.getModelProto()

        gradient_keys = list(anchors.keys())
        anchors[x] = popart.AnchorReturnType("All")

        dataFlow = popart.DataFlow(batches_per_step, anchors)

        opts = popart.SessionOptions()
        opts.executionPhaseSettings.stages = options["stages"]

        opts.executionPhaseSettings.phases = (
            options["numLayers"] * stride if options["phasedExecution"] else 0)
        opts.enableOutlining = options["outlining"]

        if "phaseSchedule" in options:
            opts.executionPhaseSettings.schedule = options["phaseSchedule"]

        # Phased execution currently does its own recompute annotations
        opts.autoRecomputation = (popart.RecomputationType.Standard
                                  if options["explicitRecomputation"] else
                                  popart.RecomputationType.NoRecompute)

        opts.outlineThreshold = -np.inf
        opts.enableOutliningCopyCostPruning = False
        opts.virtualGraphMode = (popart.VirtualGraphMode.ExecutionPhases
                                 if options["phasedExecution"] else
                                 popart.VirtualGraphMode.Manual)
        opts.explicitRecomputation = options["explicitRecomputation"]
        opts.aliasZeroCopy = options["aliasZeroCopy"]

        opts.batchSerializationSettings.factor = options["batchSerialize"]
        if "batchSchedule" in options:
            opts.batchSerializationSettings.batchSchedule = options[
                "batchSchedule"]
        if "batchConcat" in options:
            # Do not concatenate the batch across phases and virtual graphs
            # (causes more, smalle transfers but allows for individual sub-batch
            # elements to be transferred)
            opts.batchSerializationSettings.concatOnVirtualGraphChange = options[
                "batchConcat"]
            opts.batchSerializationSettings.concatOnExecutionPhaseChange = options[
                "batchConcat"]
            # Wait with loading activations until they are required
            opts.executionPhaseSettings.activationIOSchedule = popart.ExecutionPhaseIOSchedule.OnDemand

        if "tensorLocationSettings" in options and options[
                "tensorLocationSettings"]:
            opts.activationTensorLocationSettings = options[
                "tensorLocationSettings"]
            opts.weightTensorLocationSettings = options[
                "tensorLocationSettings"]
            opts.optimizerStateTensorLocationSettings = options[
                "tensorLocationSettings"]
            opts.accumulatorTensorLocationSettings = options[
                "tensorLocationSettings"]
        if "weightTensorLocationSettings" in options and options[
                "weightTensorLocationSettings"]:
            opts.weightTensorLocationSettings = options[
                "weightTensorLocationSettings"]
        if options["replication"] > 1:
            opts.replicatedGraphCount = options["replication"]
            opts.enableReplicatedGraphs = True
        if "ioTiles" in options:
            opts.numIOTiles = options["ioTiles"]

        pat = popart.Patterns(popart.PatternsLevel.Default)
        if options["phasedExecution"]:
            numIpus = options["stages"]
        else:
            numIpus = options["numLayers"] + 1
        if options["replication"] > 1:
            numIpus = numIpus * options["replication"]

        with tu.create_test_device(numIpus,
                                   pattern=popart.SyncPattern.Full) as device:

            session = popart.TrainingSession(fnModel=proto,
                                             dataFlow=dataFlow,
                                             userOptions=opts,
                                             loss=l1,
                                             optimizer=popart.ConstSGD(0.1),
                                             patterns=pat,
                                             deviceInfo=device)

            session.prepareDevice()

            session.weightsFromHost()

            anchors = session.initAnchorArrays()
            for k, v in anchors.items():
                print(f"anchor_before {k}={v.shape}")

            inputs = {x_in: input_data, mask: mask_data}
            stepio = popart.PyStepIO(inputs, anchors)

            for __ in range(10):
                session.run(stepio)

            session.modelToHost(
                str(tmpdir / f"streamingmemory_attention_{index}.onnx"))

            if options["replication"] > 1:
                for k, v in anchors.items():
                    if k in gradient_keys:
                        # The gradient anchors will have an additional replication axis.
                        anchors[k] = np.sum(v,
                                            1 if batches_per_step > 1 else 0)
                    else:
                        # Output tensor needs reshaping.
                        anchors[k] = np.reshape(anchors[k], [
                            batches_per_step, sequence_length * batch_size,
                            hidden_size
                        ])
                for k, v in anchors.items():
                    print(f"anchor_after {k}={v.shape}")

            return anchors

    test_results = []

    # AliasZeroCopy only supported with explicit recomputation, but not with
    # standard recomputation
    # Phased execution only supported with explicit recomputaton, but not with
    # standard recomputation

    test_variants = []

    defaultOffChip = popart.TensorLocationSettings(
        location=popart.TensorLocation(
            storage=popart.TensorStorage.OffChip,
            loadTileSet=popart.TileSet.Compute,
            storageTileSet=popart.TileSet.Compute,
            replicatedTensorSharding=popart.ReplicatedTensorSharding.Off),
        minElementsForOffChip=0,
        minElementsForReplicatedTensorSharding=2)

    ioOffChip = popart.TensorLocationSettings(
        location=popart.TensorLocation(
            storage=popart.TensorStorage.OffChip,
            loadTileSet=popart.TileSet.IO,
            storageTileSet=popart.TileSet.IO,
            replicatedTensorSharding=popart.ReplicatedTensorSharding.Off),
        minElementsForOffChip=0,
        minElementsForReplicatedTensorSharding=2)

    # Ground truth variant
    test_variants.append({
        "stages": 2,
        "numLayers": 3,
        "phasedExecution": False,
        "outlining": False,
        "explicitRecomputation": False,
        "aliasZeroCopy": False,
        "batchSerialize": 1,
        "replication": 1,
    })

    test_variants.append({
        "stages": 2,
        "numLayers": 3,
        "phasedExecution": False,
        "outlining": False,
        "explicitRecomputation": False,
        "aliasZeroCopy": False,
        "batchSerialize": 4,
        "replication": 1,
    })

    test_variants.append({
        "stages": 2,
        "numLayers": 3,
        "phasedExecution": True,
        "outlining": False,
        "explicitRecomputation": False,
        "aliasZeroCopy": False,
        "batchSerialize": 1,
        "replication": 1,
        "tensorLocationSettings": defaultOffChip,
    })

    test_variants.append({
        "stages": 2,
        "numLayers": 3,
        "phasedExecution": True,
        "outlining": True,
        "explicitRecomputation": False,
        "aliasZeroCopy": False,
        "batchSerialize": 1,
        "replication": 1,
        "tensorLocationSettings": defaultOffChip,
    })

    test_variants.append({
        "stages": 2,
        "numLayers": 3,
        "phasedExecution": True,
        "outlining": True,
        "explicitRecomputation": True,
        "aliasZeroCopy": False,
        "batchSerialize": 1,
        "replication": 1,
        "tensorLocationSettings": defaultOffChip,
    })

    test_variants.append({
        "stages": 2,
        "numLayers": 3,
        "phasedExecution": True,
        "outlining": True,
        "explicitRecomputation": True,
        "aliasZeroCopy": True,
        "batchSerialize": 1,
        "replication": 1,
        "tensorLocationSettings": defaultOffChip,
    })

    # Test batch serialized single device per replica execution, where all
    # streaming memory traffic goes through IO tiles, and activations are
    # stored and loaded one-by-one
    test_variants.append({
        "stages": 1,
        "stride": 4,
        "numLayers": 3,
        "phasedExecution": True,
        "outlining": True,
        "explicitRecomputation": True,
        "aliasZeroCopy": True,
        "batchSerialize": 4,
        "batchConcat": False,
        "replication": 2,
        "tensorLocationSettings": ioOffChip,
        "ioTiles": 192
    })

    # Test batch serialized single device per replica execution, where all
    # streaming memory traffic goes through IO tiles, and loading of the next
    # phase happens before storing the current phase
    test_variants.append({
        "stages": 1,
        "stride": 1,
        "numLayers": 3,
        "phasedExecution": True,
        "phaseSchedule": popart.ExecutionPhaseSchedule.BatchClusteredIO,
        "outlining": False,
        "explicitRecomputation": True,
        "aliasZeroCopy": True,
        "batchSerialize": 4,
        "batchConcat": True,
        "replication": 2,
        "tensorLocationSettings": ioOffChip,
        "ioTiles": 192
    })

    # Test a variety of batch serialisation schedules.
    for batchSchedule in [
            popart.BatchSerializationBatchSchedule.Scheduler,
            popart.BatchSerializationBatchSchedule.Isomorphic,
            popart.BatchSerializationBatchSchedule.OverlapOnIo,
            popart.BatchSerializationBatchSchedule.OverlapOnCompute,
    ]:

        test_variants.append({
            "stages": 1,
            "stride": 4,
            "numLayers": 3,
            "phasedExecution": True,
            "outlining": False,
            "explicitRecomputation": True,
            "aliasZeroCopy": True,
            "batchSerialize": 4,
            "batchSchedule": batchSchedule,
            "batchConcat": False,
            "replication": 2,
            "tensorLocationSettings": ioOffChip,
            "ioTiles": 192
        })

    # Test replicated tensor sharding + on chip (no outlining).
    test_variants.append({
        "stages":
        2,
        "numLayers":
        3,
        "phasedExecution":
        True,
        "outlining":
        False,
        "explicitRecomputation":
        False,
        "aliasZeroCopy":
        False,
        "batchSerialize":
        1,
        "replication":
        2,
        "tensorLocationSettings":
        defaultOffChip,
        "weightTensorLocationSettings":
        popart.TensorLocationSettings(location=popart.TensorLocation(
            storage=popart.TensorStorage.OnChip,
            loadTileSet=popart.TileSet.Compute,
            storageTileSet=popart.TileSet.Compute,
            replicatedTensorSharding=popart.ReplicatedTensorSharding.On),
                                      minElementsForOffChip=0,
                                      minElementsForReplicatedTensorSharding=2)
    })

    # Test replicated tensor sharding + off chip (no outlining).
    test_variants.append({
        "stages":
        2,
        "numLayers":
        3,
        "phasedExecution":
        True,
        "outlining":
        False,
        "explicitRecomputation":
        False,
        "aliasZeroCopy":
        False,
        "batchSerialize":
        1,
        "replication":
        2,
        "tensorLocationSettings":
        defaultOffChip,
        "weightTensorLocationSettings":
        popart.TensorLocationSettings(location=popart.TensorLocation(
            storage=popart.TensorStorage.OffChip,
            loadTileSet=popart.TileSet.Compute,
            storageTileSet=popart.TileSet.Compute,
            replicatedTensorSharding=popart.ReplicatedTensorSharding.On),
                                      minElementsForOffChip=0,
                                      minElementsForReplicatedTensorSharding=2)
    })

    index = 0
    for test_option in test_variants:
        print(f"Running {index}: {test_option}")
        test_results.append(run_test(index, test_option))
        index += 1

    gt_onnx = onnx.load(str(tmpdir / f"streamingmemory_attention_0.onnx"))

    for i in range(1, index):
        print(f"Testing run {i}: {test_variants[i]}")
        for key in test_results[0].keys():
            assert np.all(
                np.isclose(test_results[0][key],
                           test_results[i][key],
                           equal_nan=False))

        val_onnx = onnx.load(
            str(tmpdir / f"streamingmemory_attention_{i}.onnx"))
        for j in range(len(gt_onnx.graph.initializer)):
            print(f"Checking initializer {j}")
            gt = gt_onnx.graph.initializer[j]
            gt = numpy_helper.to_array(gt)
            val = val_onnx.graph.initializer[j]
            val = numpy_helper.to_array(val)
            assert np.allclose(gt, val, equal_nan=False)
