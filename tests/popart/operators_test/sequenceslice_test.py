# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
import torch.nn.functional as F
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu


@pytest.mark.parametrize("inplacing", [False, True])
def test_sequenceslice(op_tester, inplacing):
    def run_test(source_shape, dest_shape, N, source_offset, dest_offset):
        source = np.arange(np.prod(source_shape)) + 10
        source = np.reshape(source, source_shape).astype(np.float32)

        dest = np.zeros(dest_shape, np.float32)
        N = np.asarray(N, np.uint32)
        source_offset = np.asarray(source_offset, np.uint32)
        dest_offset = np.asarray(dest_offset, np.uint32)

        def init_builder(builder):
            source_id = builder.addInputTensor(source)
            dest_id = builder.addInputTensor(dest)
            N_id = builder.addInputTensor(N)
            source_offset_id = builder.addInputTensor(source_offset)
            dest_offset_id = builder.addInputTensor(dest_offset)
            o = builder.aiGraphcore.sequenceslice(
                [source_id, dest_id, N_id, source_offset_id, dest_offset_id],
                True)
            builder.addOutputTensor(o)
            return [o]

        def reference(ref_data):
            result = np.copy(dest)
            for i in range(N.size):
                result[dest_offset[i]:dest_offset[i] +
                       N[i]] = source[source_offset[i]:source_offset[i] + N[i]]
            return [result]

        op_tester.inplacing = inplacing
        op_tester.run(init_builder, reference, 'infer')

    run_test([6, 6], [3, 6], [1, 1, 1], [0, 2, 4], [0, 1, 2])
    run_test([6, 6], [3, 6], [1, 2], [0, 4], [2, 0])
    run_test([6, 6], [3, 6], [1, 2], [0, 4], [2, 0])
    run_test([6, 6], [4, 6], [1, 1], [0, 3], [0, 2])
