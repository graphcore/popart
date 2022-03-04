# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# Check that no error is thrown if opts.cachePath is set and the device is a
# cpu device.
def test_cpu_device(tmp_path):
    # Create a builder and construct a graph
    builder = popart.Builder()

    data_shape = popart.TensorInfo("FLOAT", [1])

    a = builder.addInputTensor(data_shape)
    b = builder.addInputTensor(data_shape)

    o = builder.aiOnnx.add([a, b])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    # Describe how to run the model
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.enableEngineCaching = True
    opts.cachePath = str(tmp_path / 'saved_graph')

    # Create a session to compile and execute the graph
    with tu.create_test_device() as device:
        session = popart.InferenceSession(fnModel=proto,
                                          dataFlow=dataFlow,
                                          userOptions=opts,
                                          deviceInfo=device)

        # Compile graph
        session.prepareDevice()

        # Create buffers to receive results from the execution
        anchors = session.initAnchorArrays()

        # Generate some random input data
        data_a = np.random.rand(1).astype(np.float32)
        data_b = np.random.rand(1).astype(np.float32)

        stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
        session.run(stepio)

        assert anchors[o] == data_a + data_b
