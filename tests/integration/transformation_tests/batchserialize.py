# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import onnx
from onnx import numpy_helper

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_weight_update(tmpdir):
    def run(model_file_name, batchSerializationFactor):
        bsize = 8
        dsize = 10
        builder = popart.Builder()
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [bsize, dsize, dsize]))

        def add_layer(in_id):
            w = builder.addInitializedInputTensor(np.ones([dsize, dsize], np.float32))
            matmul_id = builder.aiOnnx.matmul([in_id, w])
            return matmul_id

        m1 = add_layer(ip)
        m2 = add_layer(m1)
        m3 = add_layer(m2)

        out = builder.aiGraphcore.l1loss([m3], 0.1)
        builder.addOutputTensor(out)

        with tu.create_test_device(1) as device:

            dfAnchors = {}

            opts = popart.SessionOptions()
            opts.enableOutlining = True
            opts.batchSerializationSettings.factor = batchSerializationFactor

            proto = builder.getModelProto()

            session = popart.TrainingSession(
                fnModel=proto,
                dataFlow=popart.DataFlow(1, dfAnchors),
                optimizer=popart.ConstSGD(0.1),
                loss=out,
                patterns=popart.Patterns(popart.PatternsLevel.All),
                userOptions=opts,
                deviceInfo=device,
            )

            session.prepareDevice()
            session.weightsFromHost()
            anchors = session.initAnchorArrays()

            ip_data = np.ones((bsize, dsize, dsize), dtype=np.float32)
            stepio = popart.PyStepIO({ip: ip_data}, anchors)

            session.run(stepio)

            session.modelToHost(str(tmpdir / model_file_name))

    run("without_batchserial.onnx", 0)
    run("with_batchserial.onnx", 4)

    without_batchserial = onnx.load(str(tmpdir / "without_batchserial.onnx"))
    with_batchserial = onnx.load(str(tmpdir / "with_batchserial.onnx"))

    for i in range(len(without_batchserial.graph.initializer)):
        print(f"Checking initializer {i}")
        lhs = without_batchserial.graph.initializer[i]
        lhs = numpy_helper.to_array(lhs)
        rhs = with_batchserial.graph.initializer[i]
        rhs = numpy_helper.to_array(rhs)
        assert np.allclose(lhs, rhs)


"""
For tensors generated by the init Op, by default the batch axis is assumed
to be the 0th. Users can explicitly specify an axis in cases where it differs
from the default. This test checks that batch serialisation works in both
the default case and when we specify a different axis.
"""


def test_init():
    def run(transposed):
        bsize = 8
        dsize = 10
        builder = popart.Builder()
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [bsize, dsize, dsize]))
        if transposed:
            # Explicitly specify the batch dimension for init
            init = builder.aiGraphcore.init(
                [dsize, dsize, bsize], popart.DataType.FLOAT, popart.InitType.Zero, 2
            )
        else:
            init = builder.aiGraphcore.init(
                [bsize, dsize, dsize], popart.DataType.FLOAT, popart.InitType.Zero, 0
            )

        def add_layer(in_id):
            w = builder.addInitializedInputTensor(np.ones([dsize, dsize], np.float32))
            if transposed:
                inputs = [w, in_id]
            else:
                inputs = [in_id, w]
            matmul_id = builder.aiOnnx.matmul(inputs)
            return matmul_id

        if transposed:
            ip_t = builder.aiOnnx.transpose([ip])
        else:
            ip_t = ip
        m1 = add_layer(ip_t)
        init = builder.aiOnnx.add([init, m1])
        m2 = add_layer(m1)
        init = builder.aiOnnx.add([init, m2])
        m3 = add_layer(m2)
        init = builder.aiOnnx.add([init, m3])

        out = builder.aiGraphcore.l1loss([init], 0.1)
        builder.addOutputTensor(out)

        with tu.create_test_device(1) as device:

            dfAnchors = {out: popart.AnchorReturnType("All")}

            opts = popart.SessionOptions()
            opts.enableOutlining = True
            opts.batchSerializationSettings.factor = 4

            proto = builder.getModelProto()

            session = popart.InferenceSession(
                fnModel=proto,
                dataFlow=popart.DataFlow(1, dfAnchors),
                patterns=popart.Patterns(popart.PatternsLevel.All),
                userOptions=opts,
                deviceInfo=device,
            )

            session.prepareDevice()
            session.weightsFromHost()
            anchors = session.initAnchorArrays()

            ip_data = np.ones((bsize, dsize, dsize), dtype=np.float32)
            stepio = popart.PyStepIO({ip: ip_data}, anchors)

            session.run(stepio)

    # Run with batch axis 0
    run(False)
    # Run with transposed input so that the batch axis will be the last,
    # which requires explicit annotation for the init op
    run(True)


"""
In most cases the batch axis is the 0th. However for tensors in and out of LSTM
for example the batch axis is different (depending on the tensor considered).
This test checks that batch serialisation picks up the correct axis and works.
"""


def test_lstm():
    def run():
        bsize = 2
        seq_len = 5
        input_size = 3
        hidden_size = 10

        builder = popart.Builder()
        ip = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [seq_len, bsize, input_size])
        )
        weights = builder.addInitializedInputTensor(
            np.random.rand(4, input_size + hidden_size, hidden_size).astype(np.float32)
        )
        biases = builder.addInitializedInputTensor(
            np.random.rand(4, hidden_size).astype(np.float32)
        )
        initial_state = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [2, bsize, hidden_size])
        )
        Y = builder.aiGraphcore.lstm([ip, weights, biases, initial_state])[0]

        out = builder.aiGraphcore.l1loss([Y], 0.1)
        builder.addOutputTensor(out)

        with tu.create_test_device(1) as device:

            dfAnchors = {out: popart.AnchorReturnType("All")}

            opts = popart.SessionOptions()
            opts.batchSerializationSettings.factor = 2

            proto = builder.getModelProto()

            session = popart.InferenceSession(
                fnModel=proto,
                dataFlow=popart.DataFlow(1, dfAnchors),
                patterns=popart.Patterns(popart.PatternsLevel.All),
                userOptions=opts,
                deviceInfo=device,
            )

            session.prepareDevice()
            session.weightsFromHost()
            anchors = session.initAnchorArrays()

            ip_data = np.ones((seq_len, bsize, input_size), dtype=np.float32)
            initial_data = np.ones((2, bsize, hidden_size), dtype=np.float32)
            stepio = popart.PyStepIO(
                {ip: ip_data, initial_state: initial_data}, anchors
            )

            session.run(stepio)

    run()
