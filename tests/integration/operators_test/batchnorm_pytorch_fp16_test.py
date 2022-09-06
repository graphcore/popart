# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import test_util as tu


@tu.requires_ipu
def test_batchnorm_fp16_torch_version_noexception(op_tester):
    """Test using the torch version of the operator doesn't throw a FP exception"""
    """Hw only"""
    # create test data
    np.random.seed(43)
    d1 = np.random.rand(16, 64, 112, 2).astype(np.float16)
    d2 = np.random.rand(16, 64, 112, 1).astype(np.float16)

    scale = np.ones(64).astype(np.float16)
    b = np.zeros(64).astype(np.float16)
    mean = np.zeros(64).astype(np.float16)
    var = np.ones(64).astype(np.float16)
    epsilon = 0.000010

    popart_momentum = 0.9
    num_output = 5

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        w1 = builder.addInitializedInputTensor(
            (np.random.rand(2, 112)).astype(np.float16)
        )
        w2 = builder.addInitializedInputTensor(
            (np.random.rand(112, 1)).astype(np.float16)
        )
        iScale = builder.addInitializedInputTensor(scale)
        iB = builder.addInitializedInputTensor(b)
        iMean = builder.addInitializedInputTensor(mean)
        iVar = builder.addInitializedInputTensor(var)

        mm = builder.aiOnnx.matmul([i1, w1])
        builder.setPartialsType(mm, "half")
        o_y, _, _, _, _ = builder.aiGraphcore.batchnormalization(
            [mm, iScale, iB, iMean, iVar], num_output, epsilon, popart_momentum
        )
        o_y = builder.aiOnnx.matmul([o_y, w2])
        builder.setPartialsType(o_y, "half")
        lossId = builder.aiOnnx.sub([o_y, i2])
        lossId = builder.aiGraphcore.l1loss([lossId], 0.1)
        builder.addOutputTensor(lossId)

        return [lossId]

    def reference(_):
        return [np.array(2.69).astype(np.float16)]

    op_tester.atol = 1e-05
    op_tester.tilesPerIPU = tu.USE_ALL_TILES
    op_tester.options.enableFloatingPointChecks = True
    op_tester.run(init_builder, reference)
