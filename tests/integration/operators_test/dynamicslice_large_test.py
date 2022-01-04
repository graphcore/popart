# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu
def test_dynamicslice_large(op_tester):
    batch_size = 32
    data = np.random.rand(batch_size, 128, 1024).astype(np.float16)
    axes = [0]
    sizes = [1]

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        result = []
        for sliceid in range(batch_size):
            index = builder.addInputTensor(np.asarray([sliceid], np.uint32))
            out = builder.aiGraphcore.dynamicslice([tensor, index],
                                                   axes=axes,
                                                   sizes=sizes,
                                                   noOverlap=True)
            builder.addOutputTensor(out)
            result.append(out)
        return result

    def reference(ref_data):
        result = []
        for sliceid in range(batch_size):
            result.append(data[sliceid:(sliceid + 1), :])
        return result

    #op_tester.numIPUs = 1
    op_tester.device = tu.create_test_device(numIpus=1)
    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


@tu.requires_ipu
def test_dynamicslice_update_large(op_tester):
    batch_size = 32
    weight = np.random.rand(1, 128, 128).astype(np.float16)
    data = np.random.rand(batch_size, 128, 1024).astype(np.float16)
    axes = [0]
    sizes = [1]

    def init_builder(builder):
        tensor = builder.addInputTensor(data, "data")
        weight_tensor = builder.addInputTensor(weight, "weight")
        result = []
        concat = builder.aiGraphcore.init([batch_size, 128, 1024],
                                          popart.DataType.FLOAT16,
                                          popart.InitType.NoInit, "test_init")
        for sliceid in range(batch_size):
            #with builder.schedulePriority(-sliceid):
            index = builder.addInputTensor(np.asarray([sliceid], np.uint32),
                                           "SID_" + str(sliceid))
            out = builder.aiGraphcore.dynamicslice([tensor, index],
                                                   axes=axes,
                                                   sizes=sizes,
                                                   noOverlap=True)
            out = builder.aiOnnx.matmul([weight_tensor, out])
            concat = builder.aiGraphcore.dynamicupdate([concat, index, out],
                                                       axes=axes,
                                                       sizes=sizes,
                                                       noOverlap=True)
        builder.addOutputTensor(concat)
        result.append(concat)
        return result

    def reference(ref_data):
        result = []
        for sliceid in range(batch_size):
            result.append(
                np.dot(weight,
                       data[sliceid:(sliceid + 1), :, :]).squeeze(axis=2))
        return [np.concatenate(result)]

    op_tester.atol = 1e-2
    op_tester.rtol = 1e-2
    op_tester.options.enableOutlining = True
    op_tester.options.outlineThreshold = 0.0
    op_tester.options.enableOutliningCopyCostPruning = False
    #op_tester.options.aliasZeroCopy = True
    op_tester.device = tu.create_test_device(numIpus=1)
    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')
