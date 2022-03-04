# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import test_util as tu


def test_basic():
    bias_data = [np.random.rand(2).astype(np.float32) for _ in range(2)]
    filt_data = np.random.rand(2, 2, 3, 3).astype(np.float32)
    input_data = np.random.rand(1, 2, 4, 4).astype(np.float32)

    def run_test(enableOutlining):
        builder = popart.Builder()

        i1 = builder.addInputTensor("FLOAT", [1, 2, 4, 4], "data")

        i2 = builder.addInitializedInputTensor(filt_data, "filter")

        ibias0 = builder.addInitializedInputTensor(bias_data[0], "bias0")
        ibias1 = builder.addInitializedInputTensor(bias_data[1], "bias1")

        c1 = builder.aiOnnx.conv([i1, i2, ibias0],
                                 dilations=[1, 1],
                                 pads=[1, 1, 1, 1],
                                 strides=[1, 1])
        o = builder.aiOnnx.conv([c1, i2, ibias1],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                strides=[1, 1])
        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(1, {
            c1: popart.AnchorReturnType("All"),
            o: popart.AnchorReturnType("All")
        })

        opts = popart.SessionOptions()
        opts.enableOutlining = enableOutlining
        opts.enableOutliningCopyCostPruning = False

        with tu.create_test_device() as device:
            session = popart.InferenceSession(fnModel=proto,
                                              dataFlow=dataFlow,
                                              userOptions=opts,
                                              deviceInfo=device)

            session.prepareDevice()

            anchors = session.initAnchorArrays()

            inputs = {i1: input_data}
            stepio = popart.PyStepIO(inputs, anchors)

            session.run(stepio)

        return anchors[o]

    reference = run_test(enableOutlining=False)
    result = run_test(enableOutlining=True)

    assert np.array_equal(reference, result)
