# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import torch
import onnx
from onnx import numpy_helper
import test_util as tu
import json


def test_weight_update(tmpdir):
    def run(model_file_name, enableOutlining):
        dsize = 10
        ratio = 0.5
        builder = popart.Builder()
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize, dsize]))
        d__ip = popart.reservedGradientPrefix() + ip

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
            anchorIds.append(popart.reservedGradientPrefix() + i)

        out = builder.aiGraphcore.identityloss([m3])
        builder.addOutputTensor(out)

        device = tu.create_test_device()

        dfAnchors = {}
        for anchorId in anchorIds:
            dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()
        opts.enableOutlining = enableOutlining
        opts.separateCallOpPdfs = False

        proto = builder.getModelProto()

        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=popart.DataFlow(1, dfAnchors),
            optimizer=popart.ConstSGD(0.1),
            loss=out,
            patterns=popart.Patterns(popart.PatternsLevel.All),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        ip_data = np.ones((dsize, dsize), dtype=np.float32)
        stepio = popart.PyStepIO({ip: ip_data}, anchors)

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
        builder = popart.Builder()
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize, dsize]))
        d__ip = popart.reservedGradientPrefix() + ip

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
            anchorIds.append(popart.reservedGradientPrefix() + i)

        out = builder.aiGraphcore.identityloss([m3])
        builder.addOutputTensor(out)

        device = tu.create_test_device()

        dfAnchors = {}
        for anchorId in anchorIds:
            dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()
        opts.enableOutlining = enableOutlining

        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(batches_per_step, dfAnchors),
            optimizer=popart.ConstSGD(0.1),
            loss=out,
            patterns=popart.Patterns(popart.PatternsLevel.All),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        ip_data = np.ones((batches_per_step, dsize, dsize), dtype=np.float32)
        stepio = popart.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)

        return anchors

    without_outlining = run(False)
    with_outlining = run(True)

    for key in without_outlining.keys():
        print(f'Checking anchor {key}')
        lhs = without_outlining[key]
        rhs = with_outlining[key]

        assert np.allclose(lhs, rhs)


def test_outlining_in_subgraphs():
    data = [np.random.rand(4, 4).astype(np.float32) for i in range(2)]
    weights = [np.random.rand(4, 4).astype(np.float32) for i in range(2)]

    def run_popart():
        # Main graph: add, matmul, add, Call(0)
        # Subgraph 0: add, matmul
        bld = popart.Builder()
        i0 = bld.addInputTensor("FLOAT", [4, 4])
        i1 = bld.addInputTensor("FLOAT", [4, 4])
        w0 = bld.addInitializedInputTensor(weights[0])
        w1 = bld.addInitializedInputTensor(weights[1])
        x = bld.aiOnnx.add([i0, i1])
        x = bld.aiOnnx.matmul([x, w0])
        x = bld.aiOnnx.add([x, i1])

        def create_subgraph():
            subgraph_builder = bld.createSubgraphBuilder()
            tensor_info = popart.TensorInfo('FLOAT', [4, 4])
            i0 = subgraph_builder.addInputTensor(tensor_info)
            i1 = subgraph_builder.addInputTensor(tensor_info)
            i2 = subgraph_builder.addInputTensor(tensor_info)
            y = subgraph_builder.aiOnnx.add([i0, i1])
            y = subgraph_builder.aiOnnx.matmul([y, i2])
            subgraph_builder.addOutputTensor(y)
            return subgraph_builder

        x = bld.aiGraphcore.call([x, i1, w1], 1, create_subgraph())[0]

        o = x
        proto = bld.getModelProto()

        sess = popart.InferenceSession(fnModel=proto,
                                       deviceInfo=tu.create_test_device(),
                                       dataFlow=popart.DataFlow(1, [o]))
        sess.prepareDevice()

        anchors = sess.initAnchorArrays()

        stepio = popart.PyStepIO({i0: data[0], i1: data[1]}, anchors)
        sess.weightsFromHost()

        sess.run(stepio)
        return anchors[o], json.loads(
            sess._serializeIr(popart.IrSerializationFormat.JSON))

    def run_reference():
        x = data[0] + data[1]
        x = np.matmul(x, weights[0])
        x = x + data[1]
        x = x + data[1]
        x = np.matmul(x, weights[1])
        return x

    popart_output, ir = run_popart()
    ref_output = run_reference()
    print(f'popart_output: {popart_output}\n')
    print(f'ref_output: {ref_output}')
    assert np.allclose(popart_output, ref_output)

    ops = [j['type'] for i in ir.values() for j in i]

    # Outlining should have added another subgraph.
    graph_count = len(ir)
    assert graph_count == 3

    # The 2 MatMul ops should have been outlined.
    assert ops.count('MatMul') == 1

    # 2 Call ops should have been added for the outlined MatMuls.
    assert ops.count('Call') == 3
