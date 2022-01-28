# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import json
import numpy as np
import popart
import torch
import pytest
import tempfile
import pva

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu


# Demonstrates a situation in which TensorRemapOp can reduce total memory
# by remapping the tensor once for 5 identical consumers (matmuls)
def test_tensorremap(op_tester):
    num_matmuls = 5
    size = 128
    axis = 1

    expand_shape = np.array([2, 16, size, size]).astype(np.int32)

    v = np.random.rand(2, 16, size, size).astype(np.float32)

    # Weights
    ws = [
        np.random.rand(2, 16, size, size).astype(np.float32)
        for i in range(num_matmuls)
    ]

    # Bias
    bs = np.random.rand(size, size).astype(np.int32)

    # Runs the test and calculates total tile memory by using profiling (PVA)
    def get_memory(remap):
        def init_builder(builder):
            x = builder.addInitializedInputTensor(v)
            wst = [builder.addInitializedInputTensor(w) for w in ws]
            b = builder.addInitializedInputTensor(bs)
            eshape = builder.aiOnnx.constant(expand_shape)

            b = builder.aiOnnx.expand([b, eshape])
            x = builder.aiOnnx.cast([b], "FLOAT")
            if remap:
                b = builder.aiOnnx.expand([b, eshape])
                b = builder.aiGraphcore.tensorremap([b])

            b = builder.aiOnnx.cast([b], "FLOAT")

            for i in range(num_matmuls):
                w = wst[i]
                x = builder.aiOnnx.matmul([x, w])
                x = builder.aiOnnx.add([x, b])

            y = builder.aiOnnx.reducesum([x], keepdims=0)

            builder.addOutputTensor(y)
            return ([y])

        def reference(ref_data):
            wst = [torch.tensor(w, requires_grad=True) for w in ws]
            b = torch.tensor(bs.astype(np.float32), requires_grad=True)

            x = b.reshape([1, 1, size, size]).expand(2, 16, -1, -1)

            for i in range(num_matmuls):
                w = wst[i]
                x = torch.matmul(x, w) + b

            y = torch.sum(x)

            return ([y])

        tempDir = tempfile.TemporaryDirectory()
        op_tester.numIPUs = 1
        op_tester.tilesPerIPU = 64
        op_tester.setPatterns(popart.PatternsLevel.Default,
                              enableRuntimeAsserts=False)
        op_tester.options.enableOutlining = True
        op_tester.options.virtualGraphMode = popart.VirtualGraphMode.Auto
        op_tester.options.engineOptions["autoReport.directory"] = tempDir.name
        op_tester.options.engineOptions[
            "autoReport.outputGraphProfile"] = "true"

        session = op_tester.run(init_builder, reference, 'infer')

        report = session.getReport()

        total_mem = 0
        for t in report.compilation.tiles:
            total_mem = total_mem + t.memory.total.includingGaps
        print(f'total_mem: {total_mem}')
        return total_mem

    tm0 = get_memory(False)  # Reference value: 28326188
    tm1 = get_memory(True)  # Reference value: 28270760
    assert (tm0 > tm1)
