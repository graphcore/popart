import popart
import numpy as np
import torch
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
        ip = builder.addInputTensor(
            popart.TensorInfo("FLOAT", [bsize, dsize, dsize]))
        d__ip = popart.reservedGradientPrefix() + ip

        def add_layer(in_id):
            w = builder.addInitializedInputTensor(
                np.ones([dsize, dsize], np.float32))
            matmul_id = builder.aiOnnx.matmul([in_id, w])
            return matmul_id

        m1 = add_layer(ip)
        m2 = add_layer(m1)
        m3 = add_layer(m2)

        out = m3
        builder.addOutputTensor(out)

        device = tu.create_test_device(1)

        dfAnchors = {}

        opts = popart.SessionOptions()
        opts.enableOutlining = True
        opts.batchSerializationFactor = batchSerializationFactor

        proto = builder.getModelProto()

        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=popart.DataFlow(1, dfAnchors),
            optimizer=popart.ConstSGD(0.1),
            losses=[popart.IdentityLoss(out, "idLossVal")],
            patterns=popart.Patterns(popart.PatternsLevel.All),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        ip_data = np.ones((bsize, dsize, dsize), dtype=np.float32)
        stepio = popart.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)

        session.modelToHost(str(tmpdir / model_file_name))

    run('without_batchserial.onnx', 0)
    run('with_batchserial.onnx', 4)

    without_batchserial = onnx.load(str(tmpdir / 'without_batchserial.onnx'))
    with_batchserial = onnx.load(str(tmpdir / 'with_batchserial.onnx'))

    for i in range(len(without_batchserial.graph.initializer)):
        print(f'Checking initializer {i}')
        lhs = without_batchserial.graph.initializer[i]
        lhs = numpy_helper.to_array(lhs)
        rhs = with_batchserial.graph.initializer[i]
        rhs = numpy_helper.to_array(rhs)
        assert np.allclose(lhs, rhs)
