import poponnx
import numpy as np
import torch
import onnx
from onnx import numpy_helper


def test_weight_update(tmpdir):
    def run(model_file_name, enableOutlining):
        dsize = 10
        ratio = 0.5
        builder = poponnx.Builder()
        ip = builder.addInputTensor(
            poponnx.TensorInfo("FLOAT", [dsize, dsize]))
        d__ip = poponnx.reservedGradientPrefix() + ip

        def add_layer(in_id):
            w = builder.addInitializedInputTensor(
                np.ones([dsize, dsize], np.float32))
            matmul_id = builder.aiOnnx.matmul([in_id, w])
            return matmul_id

        m1 = add_layer(ip)
        m2 = add_layer(m1)
        m3 = add_layer(m2)

        anchorIds = []
        for i in (ip, m1, m2, m3):
            anchorIds.append(poponnx.reservedGradientPrefix() + i)

        out = m3
        builder.addOutputTensor(out)

        device = poponnx.DeviceManager().createCpuDevice()

        dfAnchors = {}
        for anchorId in anchorIds:
            dfAnchors.update({anchorId: poponnx.AnchorReturnType("ALL")})

        opts = poponnx.SessionOptions()
        opts.enableOutlining = enableOutlining
        opts.separateCallOpPdfs = False

        proto = builder.getModelProto()

        session = poponnx.TrainingSession(
            fnModel=proto,
            dataFeed=poponnx.DataFlow(1, dfAnchors),
            optimizer=poponnx.ConstSGD(0.1),
            losses=[poponnx.L1Loss(out, "l1LossVal", 0.1)],
            passes=poponnx.Patterns(poponnx.PatternsLevel.ALL),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        ip_data = np.ones((dsize, dsize), dtype=np.float32)
        stepio = poponnx.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)

        session.modelToHost(str(tmpdir / model_file_name))

    run('without_outlining.onnx', False)
    run('with_outlining.onnx', True)

    with_outlining = onnx.load(str(tmpdir / 'without_outlining.onnx'))
    without_outlining = onnx.load(str(tmpdir / 'with_outlining.onnx'))

    for i in range(len(without_outlining.graph.initializer)):
        print(f'Checking initializer {i}')
        lhs = without_outlining.graph.initializer[i]
        lhs = numpy_helper.to_array(lhs)
        rhs = with_outlining.graph.initializer[i]
        rhs = numpy_helper.to_array(rhs)
        assert np.allclose(lhs, rhs)


def test_batches_per_step_greater_than_one():
    def run(enableOutlining):
        dsize = 10
        ratio = 0.5
        batches_per_step = 2
        builder = poponnx.Builder()
        ip = builder.addInputTensor(
            poponnx.TensorInfo("FLOAT", [dsize, dsize]))
        d__ip = poponnx.reservedGradientPrefix() + ip

        def add_layer(in_id):
            w = builder.addInitializedInputTensor(
                np.ones([dsize, dsize], np.float32))
            # w = builder.aiGraphcore.printtensor([w])
            matmul_id = builder.aiOnnx.matmul([in_id, w])
            return matmul_id

        m1 = add_layer(ip)
        m2 = add_layer(m1)
        m3 = add_layer(m2)

        anchorIds = []
        for i in (ip, m1, m2, m3):
            anchorIds.append(poponnx.reservedGradientPrefix() + i)

        out = m3
        builder.addOutputTensor(out)

        device = poponnx.DeviceManager().createCpuDevice()

        dfAnchors = {}
        for anchorId in anchorIds:
            dfAnchors.update({anchorId: poponnx.AnchorReturnType("ALL")})

        opts = poponnx.SessionOptions()
        opts.enableOutlining = enableOutlining

        session = poponnx.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=poponnx.DataFlow(batches_per_step, dfAnchors),
            optimizer=poponnx.ConstSGD(0.1),
            losses=[poponnx.L1Loss(out, "l1LossVal", 0.1)],
            passes=poponnx.Patterns(poponnx.PatternsLevel.ALL),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        ip_data = np.ones((batches_per_step, dsize, dsize), dtype=np.float32)
        stepio = poponnx.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)

        return anchors

    without_outlining = run(False)
    with_outlining = run(True)

    for key in without_outlining.keys():
        print(f'Checking anchor {key}')
        lhs = without_outlining[key]
        rhs = with_outlining[key]

        assert np.allclose(lhs, rhs)
