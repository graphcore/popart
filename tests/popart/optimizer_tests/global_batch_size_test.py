# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import popart
import test_util as tu
import pprint
import json
import onnx
from onnx import numpy_helper


@tu.requires_ipu
@pytest.mark.parametrize("reduction_type", ["Sum", "Mean"])
@pytest.mark.parametrize("loss_type", ["Identity", "L1", "NLL"])
@pytest.mark.parametrize("batchserial", ["Unroll", "Loop"])
def test_global_batch_size_correctness_test_sgd_(tmpdir, reduction_type,
                                                 loss_type, batchserial):
    run_global_batch_size_correctness_test(tmpdir, reduction_type, loss_type,
                                           "SGD", batchserial)


@tu.requires_ipu
@pytest.mark.parametrize("reduction_type", ["Sum", "Mean"])
@pytest.mark.parametrize("loss_type", ["Identity", "L1", "NLL"])
@pytest.mark.parametrize("batchserial", ["Unroll", "Loop"])
def test_global_batch_size_correctness_test_sgdm(tmpdir, reduction_type,
                                                 loss_type, batchserial):
    run_global_batch_size_correctness_test(tmpdir, reduction_type, loss_type,
                                           "SGDM", batchserial)


@tu.requires_ipu
@pytest.mark.parametrize("reduction_type", ["Sum", "Mean"])
@pytest.mark.parametrize("loss_type", ["Identity", "L1", "NLL"])
@pytest.mark.parametrize("batchserial", ["Unroll", "Loop"])
def test_global_batch_size_correctness_test_adam(tmpdir, reduction_type,
                                                 loss_type, batchserial):
    run_global_batch_size_correctness_test(tmpdir, reduction_type, loss_type,
                                           "ADAM", batchserial)


@tu.requires_ipu
@pytest.mark.parametrize("reduction_type", ["Sum", "Mean"])
@pytest.mark.parametrize("loss_type", ["Identity", "L1", "NLL"])
@pytest.mark.parametrize("batchserial", ["Unroll", "Loop"])
def test_global_batch_size_correctness_test_lamb(tmpdir, reduction_type,
                                                 loss_type, batchserial):
    run_global_batch_size_correctness_test(tmpdir, reduction_type, loss_type,
                                           "LAMB", batchserial)


def run_global_batch_size_correctness_test(tmpdir, reduction_type, loss_type,
                                           optim, batchserial):
    batches_per_step = 2
    hidden_size = 16
    reduction = popart.ReductionType.Sum if reduction_type == "Sum" else popart.ReductionType.Mean

    def model(compute_batch, batch_serialization_factor, accumulation_factor,
              replication_factor):
        global_batch = compute_batch * batch_serialization_factor * accumulation_factor * replication_factor
        np.random.seed(1984)
        input_data = np.random.rand(batches_per_step, global_batch,
                                    hidden_size).astype(np.float32)
        weight_data = np.random.rand(hidden_size,
                                     hidden_size).astype(np.float32)

        builder = popart.Builder()

        d0 = builder.addInputTensor(
            popart.TensorInfo(
                'FLOAT',
                (compute_batch * batch_serialization_factor, hidden_size)),
            'data0')
        data = {
            d0:
            input_data.reshape(
                (batches_per_step, replication_factor, accumulation_factor,
                 batch_serialization_factor * compute_batch, -1))
        }

        w0 = builder.addInitializedInputTensor(weight_data, 'weight0')
        x = builder.aiOnnx.matmul([d0, w0])

        if loss_type == "L1":
            loss = builder.aiGraphcore.l1loss([x],
                                              0.1,
                                              reduction=reduction,
                                              debugPrefix='loss')
        elif loss_type == "Identity":
            loss = builder.aiGraphcore.identityloss([x],
                                                    reduction=reduction,
                                                    debugPrefix='loss')
        elif loss_type == "NLL":
            x = builder.aiOnnx.softmax([x])
            l0 = builder.addInputTensor(
                popart.TensorInfo(
                    'UINT32', (batch_serialization_factor * compute_batch, )),
                'data0')
            data[l0] = np.random.randint(0, hidden_size, size=batches_per_step * global_batch)\
                        .reshape((batches_per_step,
                            replication_factor,
                            accumulation_factor,
                            batch_serialization_factor * compute_batch,
                            -1))\
                        .astype(np.uint32)
            loss = builder.aiGraphcore.nllloss([x, l0],
                                               reduction=reduction,
                                               debugPrefix='loss')
        return builder.getModelProto(), data, [x, loss], loss

    def run_test(compute_batch, batch_serialization_factor,
                 accumulation_factor, replication_factor):
        proto, data, xs, loss = model(compute_batch,
                                      batch_serialization_factor,
                                      accumulation_factor, replication_factor)

        options = popart.SessionOptions()
        patterns = popart.Patterns(popart.PatternsLevel.All)

        if optim is "SGD":
            optimizer = popart.SGD({
                "defaultLearningRate": (0.1, False),
                "lossScaling": (20, False)
            })
        elif optim is "SGDM":
            optimizer = popart.SGD({
                "defaultLearningRate": (0.1, False),
                "defaultMomentum": (0.9, False),
                "defaultDampening": (0.1, False),  # to increase errors
                "lossScaling": (20, False),
            })
        elif optim is "ADAM":
            optimizer = popart.Adam(
                {
                    "defaultLearningRate": (0.1, False),
                    "defaultBeta1": (0.9, False),
                    "defaultBeta2": (0.999, False),
                    "lossScaling": (20, False),
                },
                mode=popart.AdamMode.AdamNoBias)  # to increase errors
        elif optim is "LAMB":
            optimizer = popart.Adam(
                {
                    "defaultLearningRate": (0.1, False),
                    "defaultBeta1": (0.9, False),
                    "defaultBeta2": (0.999, False),
                    "lossScaling": (20, False),
                },
                mode=popart.AdamMode.LambNoBias)  # to increase errors

        options.batchSerializationSettings.factor = batch_serialization_factor

        if batch_serialization_factor > 1 and batchserial == "Loop":
            options.batchSerializationSettings.method = popart.BatchSerializationMethod.Loop
            options.batchSerializationSettings.transformContext = popart.BatchSerializationTransformContext.Bwd

        if accumulation_factor > 1:
            options.enableGradientAccumulation = True
            options.accumulationFactor = accumulation_factor
            options.accumulationReductionType = reduction
        if replication_factor > 1:
            options.enableReplicatedGraphs = True
            options.replicatedGraphCount = replication_factor

        device = tu.create_test_device(replication_factor,
                                       pattern=popart.SyncPattern.Full)

        dataFlow = popart.DataFlow(
            batches_per_step, {x: popart.AnchorReturnType("ALL")
                               for x in xs})

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

        file_path = str(tmpdir / f"model_test.onnx")
        session.modelToHost(file_path)
        post_proto = onnx.load(file_path)

        device.detach()

        return [anchors[x] for x in xs], post_proto

    baseline_outputs, baseline_proto = run_test(16, 1, 1, 1)

    loss_fn = np.mean if reduction_type == "Mean" else np.sum
    baseline_loss = loss_fn(baseline_outputs[-1])

    tests = [
        # TODO T33333: tests with graph replication hang intermittently
        # run_test(4, 1, 1, 4),
        run_test(4, 1, 4, 1),
        # run_test(4, 1, 2, 2),
        run_test(4, 4, 1, 1),
        run_test(4, 2, 2, 1),
        # run_test(2, 2, 2, 2)
    ]

    rtol = 1.e-5
    atol = 1.e-5

    for i, results in enumerate(tests):
        print(f"Checking results of test {i}")
        outputs, proto = results
        assert np.allclose(outputs[0].flatten(),
                           baseline_outputs[0].flatten(),
                           rtol=rtol,
                           atol=atol)

        loss = loss_fn(outputs[-1])
        print(loss)
        assert np.allclose(loss, baseline_loss, rtol=rtol, atol=atol)

        for j in range(len(baseline_proto.graph.initializer)):
            gt = baseline_proto.graph.initializer[j]
            print(f"Checking initializer {gt.name}")
            if "Step" in gt.name:
                continue
            gt = numpy_helper.to_array(gt)
            val = proto.graph.initializer[j]
            val = numpy_helper.to_array(val)
            assert np.allclose(val, gt, rtol=rtol, atol=atol)
