# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import random

import popart
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

from packed_data_test import gen_packed_sequences, unpack, pack


@tu.requires_ipu_model
def test_multi_ipu(op_tester):
    np.random.seed(0)

    sequenceLengths = [3, 5, 7, 4, 6, 2]
    data, sequenceOffsets = gen_packed_sequences(sequenceLengths, [5])
    data = (data * 9 + 1).astype(np.uint32).astype(np.float32)

    sequenceLengths = np.array(sequenceLengths).astype(np.uint32)
    sequenceOffsets = np.array(sequenceOffsets).astype(np.uint32)

    maxSequenceLength = 10

    weight = np.random.rand(maxSequenceLength, maxSequenceLength)
    weight = (weight * 10).astype(np.uint32)

    def init_builder(builder):
        dataId = builder.addInputTensor(data, "data")
        sequenceLengthsId = builder.addInputTensor(sequenceLengths, "lengths")
        sequenceOffsetsId = builder.addInputTensor(sequenceOffsets, "offsets")

        subgraph_builder = builder.createSubgraphBuilder()

        with subgraph_builder.virtualGraph(0):
            sgi0 = subgraph_builder.addUntypedInputTensor()

            dt = subgraph_builder.aiOnnx.transpose([sgi0], [0, 2, 1])
            out = subgraph_builder.aiOnnx.matmul([sgi0, dt])

        subgraph_builder.addOutputTensor(out)

        with builder.virtualGraph(0):
            x = builder.aiGraphcore.packedDataBlock([
                dataId, sequenceOffsetsId, sequenceLengthsId,
                sequenceOffsetsId, sequenceLengthsId
            ], [maxSequenceLength], data.shape[0], 1, subgraph_builder)

        subgraph_builder = builder.createSubgraphBuilder()

        with subgraph_builder.virtualGraph(1):
            sgi0 = subgraph_builder.addUntypedInputTensor()

            out = subgraph_builder.aiOnnx.matmul([sgi0, sgi0])

        subgraph_builder.addOutputTensor(out)

        with builder.virtualGraph(1):
            out = builder.aiGraphcore.packedDataBlock([
                x, sequenceOffsetsId, sequenceLengthsId, sequenceOffsetsId,
                sequenceLengthsId
            ], [maxSequenceLength], data.shape[0], 1, subgraph_builder)

        builder.addOutputTensor(out)
        return [out]

    def reference(ref_data):
        d = unpack(data, sequenceOffsets, sequenceLengths, maxSequenceLength)

        # Stage 1
        dt = np.transpose(d, [0, 2, 1])
        mm = np.matmul(d, dt)
        # Stage 2
        mm = np.matmul(mm, mm)

        result = np.zeros([27, 10]).astype(np.float32)
        pack(mm, result, sequenceOffsets, sequenceLengths)
        return [result]

    op_tester.numIPUs = 2
    op_tester.options.virtualGraphMode = popart.VirtualGraphMode.Manual
    op_tester.options.autoRecomputation = popart.RecomputationType.Standard
    op_tester.patterns.enablePattern("PackedDataBlock", True)
    op_tester.run(init_builder, reference, 'infer')
