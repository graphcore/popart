import numpy as np
from operators_test.ipu_op_tester import ipu_op_tester


def test_gather(ipu_op_tester):
    d1 = np.random.rand(512, 4800).astype(np.float32)
    d2 = np.array([0, 1]).astype(np.int32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInitializedInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.gather([i1, i2], axis)
        builder.virtualGraph(o, 0)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    ipu_op_tester.passes = ['PreUniRepl', 'SplitGather']
    ipu_op_tester.run(init_builder, reference, 'infer')
