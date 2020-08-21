# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu
import pprint
import json
import onnx
from onnx import numpy_helper


@pytest.mark.parametrize('pipeline', [True, False])
def test_outlining_accumulation_context(pipeline, tmpdir):
    def model():
        np.random.seed(1984)
        input_data = np.random.rand(4, 2, 2, 20).astype(np.float32)
        weight_data = np.random.rand(20, 20).astype(np.float32)

        builder = popart.Builder()

        d0 = builder.addInputTensor(popart.TensorInfo('FLOAT', (2, 20)),
                                    'data0')
        x = d0

        for i in [0, 1]:
            with builder.virtualGraph(i):
                w0 = builder.addInitializedInputTensor(weight_data, 'weight0')
                x = builder.aiOnnx.matmul([x, w0])

                w1 = builder.addInitializedInputTensor(weight_data, 'weight1')
                x = builder.aiOnnx.matmul([x, w1])

                w2 = builder.addInitializedInputTensor(weight_data, 'weight2')
                x = builder.aiOnnx.matmul([x, w2])

                w3 = builder.addInitializedInputTensor(weight_data, 'weight3')
                x = builder.aiOnnx.matmul([x, w3])

        with builder.virtualGraph(i):
            loss = builder.aiGraphcore.l1loss([x], 0.1, debugPrefix='loss')

        return builder.getModelProto(), {d0: input_data}, x, loss

    def run_test(outlining):
        proto, data, x, loss = model()

        options = popart.SessionOptions()
        patterns = popart.Patterns()

        optimizer = popart.SGD({
            "defaultLearningRate": (0.1, True),
        })

        options.enableOutlining = outlining
        options.outlineThreshold = 10.0
        options.enableGradientAccumulation = True
        options.accumulationFactor = 4
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = 2
        options.virtualGraphMode = popart.VirtualGraphMode.Manual
        if pipeline:
            options.enablePipelining = True
            options.autoRecomputation = popart.RecomputationType.Pipeline

        device = tu.create_test_device(4)

        dataFlow = popart.DataFlow(1, {x: popart.AnchorReturnType("ALL")})

        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=dataFlow,
                                         userOptions=options,
                                         loss=loss,
                                         optimizer=optimizer,
                                         patterns=patterns,
                                         deviceInfo=device)

        session.prepareDevice()

        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        stepio = popart.PyStepIO(data, anchors)

        session.run(stepio)

        file_path = str(tmpdir / f"outlining_execution_context_model.onnx")
        session.modelToHost(file_path)
        post_proto = onnx.load(file_path)

        device.detach()

        graph_report = json.loads(session.getGraphReport())
        max_tile_memory = max(graph_report["memory"]["byTile"]["total"])
        total_memory = np.sum(graph_report["memory"]["byTile"]["total"])
        return session, anchors[x], post_proto, total_memory

    _, outputs_1, proto_1, total_memory_1 = run_test(False)
    sess, outputs_2, proto_2, total_memory_2 = run_test(True)

    ir = json.loads(sess._serializeIr(popart.IrSerializationFormat.JSON))

    subgraphs = [v for k, v in ir.items() if k != "maingraph"]

    # Check ReplicatedAllReduce is in a subgraph
    has_replicated_all_reduce = map(
        lambda graph: next(
            filter(lambda op: "ReplicatedAllReduce" in op["type"], graph),
            False), subgraphs)
    assert any(has_replicated_all_reduce)

    assert np.allclose(outputs_1, outputs_2)

    for j in range(len(proto_1.graph.initializer)):
        print(f"Checking initializer {j}")
        gt = proto_1.graph.initializer[j]
        gt = numpy_helper.to_array(gt)
        val = proto_2.graph.initializer[j]
        val = numpy_helper.to_array(val)
        assert np.allclose(gt, val)

    assert total_memory_1 > total_memory_2