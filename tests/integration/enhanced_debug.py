# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu
import pprint
import json


def test_basic(tmpdir):
    bias_data = [np.random.rand(2).astype(np.float32) for _ in range(2)]
    filt_data = np.random.rand(2, 2, 3, 3).astype(np.float32)
    input_data = np.random.rand(1, 2, 4, 4).astype(np.float32)

    filename = str(tmpdir) + "/debug.json"
    popart.initializePoplarDebugInfo(filename)

    def run_conv_test(enableOutlining):

        builder = popart.Builder()

        i1 = builder.addInputTensor("FLOAT", [1, 2, 4, 4], "data")

        i2 = builder.addInitializedInputTensor(filt_data, "filter")

        ibias0 = builder.addInitializedInputTensor(bias_data[0], "bias0")
        ibias1 = builder.addInitializedInputTensor(bias_data[1], "bias1")

        # Text both versions of the api
        c1 = builder.aiOnnx.conv([i1, i2, ibias0],
                                 dilations=[1, 1],
                                 pads=[1, 1, 1, 1],
                                 strides=[1, 1],
                                 debugContext="conv1")
        o = builder.aiOnnx.conv([c1, i2, ibias1],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                strides=[1, 1],
                                debugContext="conv2")
        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(1, {
            c1: popart.AnchorReturnType("All"),
            o: popart.AnchorReturnType("All")
        })

        opts = popart.SessionOptions()
        opts.enableOutlining = enableOutlining
        opts.enableOutliningCopyCostPruning = False

        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=tu.create_test_device())

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {i1: input_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        return anchors[o]

    reference = run_conv_test(enableOutlining=False)

    popart.closePoplarDebugInfo()
    with open(filename) as json_file:
        data = json.load(json_file)

        print(data)

        # Expect more that 40 contexts in this example. Exact value may change over time
        assert (len(data["contexts"]) > 40)

        # What else makes sense to test without making this test brittle.
        # Verify there are contexts appear for each of the popart api calls above
        dataFound = False
        filterFound = False
        conv1Found = False
        conv2Found = False

        for c in data["contexts"]:
            name = data["stringTable"][int(c["name"])]
            if name == "data":
                dataFound = True
            elif name == "filter":
                filterFound = True
            elif name == "conv1":
                conv1Found = True
            elif name == "conv2":
                conv2Found = True

        assert (dataFound)
        assert (filterFound)
        assert (conv1Found)
        assert (conv2Found)
