# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import onnx
from onnx import mapping
import test_util as tu


def getModelProto():
    """
    Create a simple model:

    in0 - Matmul - t0 - Relu - t1 - Batchnorm - t2 - Conv -- t3 -- NllLoss -- loss
    w0  --'                                          /             /
    w1  ---------------------------------------------       label -

    whose graph after auto-grad will contain tensors whose statistics the
    automatic loss scale transform will track to adjust the loss scale factor
    """
    builder = popart.Builder()

    t_shape = [2, 4, 2, 2]
    t0 = builder.addInputTensor("FLOAT16", t_shape)
    t_data = np.random.rand(*t_shape).astype(np.float16)
    t1 = builder.addInitializedInputTensor(t_data)
    t2 = builder.addInitializedInputTensor(t_data)

    with builder.virtualGraph(0), builder.pipelineStage(0):
        mm = builder.aiOnnx.matmul([t0, t1])
        r = builder.aiOnnx.relu([mm])

        bn_data = np.random.rand(4).astype(np.float16)
        bias = builder.addInitializedInputTensor(bn_data)
        scale = builder.addInitializedInputTensor(bn_data)
        mean = builder.addInitializedInputTensor(bn_data)
        var = builder.addInitializedInputTensor(bn_data)
        bn, _, _, _, _ = builder.aiOnnx.batchnormalization(
            [r, scale, bias, mean, var], 5, 1e-05, 0.1
        )

    with builder.virtualGraph(1), builder.pipelineStage(1):
        conv = builder.aiOnnx.conv([bn, t2])
        rs = builder.reshape_const(builder.aiOnnx, [conv], [2, 2])
        sf = builder.aiOnnx.softmax([rs])
        label_shape = [2]
        labels = builder.addInputTensor("INT32", label_shape)
        loss = builder.aiGraphcore.nllloss([sf, labels], ignoreIndex=1)

    # Pick some weight tensors that we want to set specific optimizer tensor
    # values for
    specifics = [scale, bias]

    return loss, builder.getModelProto(), t0, t_shape, labels, label_shape, specifics


def compare_weights(session0, session1, tmpdir):
    ref_path = str(tmpdir / "ref_session.onnx")
    session0.modelToHost(ref_path)
    session0_proto = onnx.load(ref_path)
    session0_weights = {}
    session1_weights = {}
    for i in range(len(session0_proto.graph.initializer)):
        init = session0_proto.graph.initializer[i]
        dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]
        empty_init = np.empty(shape=init.dims, dtype=dtype)
        session0_weights[init.name] = empty_init
        session1_weights[init.name] = empty_init

    session0.weightsToHost()
    session0.readWeights(popart.PyWeightsIO(session0_weights))
    session1.weightsToHost()
    session1.readWeights(popart.PyWeightsIO(session1_weights))

    for i in range(len(session0_proto.graph.initializer)):
        init_name = session0_proto.graph.initializer[i].name
        print("Comparing ", init_name)
        print(session0_weights[init_name])
        print(session1_weights[init_name])
        assert np.array_equal(session0_weights[init_name], session1_weights[init_name])


def run_automatic_loss_scaling_comparison_test(
    tmpdir,
    optimizer,
    shard=False,
    pipeline=False,
    replicate=False,
    grad_accumulate=False,
    accumulation_factor=3,
    bps=4,
    update_period=None,
    expected_loss_scale=[],
    toTrackTensors=None,
):
    """
    An integration test: verify that the weight updates computed by a session
    with auto loss scaling (ALS) enabled are identical to those with ALS
    disabled.
    """
    loss, proto, t0, t_shape, label, label_shape, specifics = getModelProto()
    step_size = bps
    if replicate:
        replicas = 2
        step_size *= replicas
    if grad_accumulate:
        step_size *= accumulation_factor

    opts = popart.SessionOptions()

    num_ipus = 1
    if shard:
        num_ipus = 2
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    if pipeline:
        num_ipus = 2
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.enablePipelining = True
    if replicate:
        num_ipus *= replicas
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = replicas
    if grad_accumulate:
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = accumulation_factor

    init_optimizer, new_optimizer = optimizer
    init_ls = init_optimizer.getLossScalingVal()
    for specific in specifics:
        init_optimizer.insertSpecific(specific, {"weightDecay": (0.01, False)})
        new_optimizer.insertSpecific(specific, {"weightDecay": (0.01, False)})

    with tu.create_test_device(num_ipus) as ref_device:
        with tu.create_test_device(num_ipus) as als_device:
            ref_session = popart.TrainingSession(
                fnModel=proto,
                deviceInfo=ref_device,
                dataFlow=popart.DataFlow(bps, []),
                loss=loss,
                optimizer=init_optimizer,
                userOptions=opts,
            )
            ref_session.prepareDevice()
            ref_session.weightsFromHost()
            ref_anchors = ref_session.initAnchorArrays()

            opts.automaticLossScalingSettings.enabled = True
            opts.automaticLossScalingSettings.binEdgeLocation = 0.5
            opts.automaticLossScalingSettings.thresholdUpperCountProportion = 0.2
            if toTrackTensors is not None:
                opts.automaticLossScalingSettings.toTrackTensors = toTrackTensors
                opts.automaticLossScalingSettings.gradientTensorTrackingMethod = (
                    popart.GradientTensorTrackingMethod.GradientsOfUserSpecifiedTensors
                )
            if update_period is not None:
                opts.automaticLossScalingSettings.updatePeriod = update_period

            ls_id = "finalLossScale"

            als_session = popart.TrainingSession(
                fnModel=proto,
                deviceInfo=als_device,
                dataFlow=popart.DataFlow(bps, [ls_id]),
                loss=loss,
                optimizer=init_optimizer,
                userOptions=opts,
            )
            als_session.prepareDevice()
            als_session.weightsFromHost()
            als_anchors = als_session.initAnchorArrays()

            t0_data = np.random.rand(step_size, *t_shape).astype(np.float16)
            label_data = np.random.randint(
                0, label_shape[0], step_size * label_shape[0]
            ).astype(np.int32)
            inputs = {t0: t0_data, label: label_data}

            # Run once
            ref_session.run(popart.PyStepIO(inputs, ref_anchors))
            als_session.run(popart.PyStepIO(inputs, als_anchors))

            # Verify that the loss scale has changed from its initial value
            print("Loss Scale:", als_anchors[ls_id].flatten())
            if update_period is not None:
                assert np.allclose(expected_loss_scale, als_anchors[ls_id].flatten())

            if grad_accumulate:
                updated_loss_scales = [als_anchors[ls_id].flatten()[-1]][1:-1]
            else:
                updated_loss_scales = als_anchors[ls_id].flatten()[1:-1]
            for ls in updated_loss_scales:
                assert ls != init_ls

            # Update the optimizer
            ref_session.updateOptimizerFromHost(new_optimizer)
            als_session.updateOptimizerFromHost(new_optimizer)

            # Run a second time
            ref_session.run(popart.PyStepIO(inputs, ref_anchors))
            als_session.run(popart.PyStepIO(inputs, als_anchors))

            # Verify that the weight updates are identitcal for a.l.s vs a reference
            # session
            compare_weights(ref_session, als_session, tmpdir)


def getOptimizers():
    optimizers = []

    # SGD
    sgd0 = popart.SGD(
        {
            "lossScaling": (10.0, False),
            "defaultMomentum": (0.5, False),
            "defaultVelocityScaling": (0.5, False),
            "defaultDampening": (0.5, False),
            "defaultWeightDecay": (0.5, False),
        }
    )
    sgd1 = popart.SGD(
        {
            "lossScaling": (0.2, False),
            "defaultMomentum": (0.2, False),
            "defaultVelocityScaling": (0.2, False),
            "defaultDampening": (0.2, False),
            "defaultWeightDecay": (0.2, False),
        }
    )
    optimizers.append([sgd0, sgd1])

    # Adam
    adam0 = popart.Adam(
        {
            "lossScaling": (10.0, False),
            "defaultLearningRate": (0.5, False),
            "defaultWeightDecay": (0.5, False),
            "defaultBeta1": (0.5, False),
            "defaultBeta2": (0.5, False),
            "defaultEps": (0.5, False),
        }
    )
    adam1 = popart.Adam(
        {
            "lossScaling": (0.2, False),
            "defaultLearningRate": (0.2, False),
            "defaultWeightDecay": (0.2, False),
            "defaultBeta1": (0.2, False),
            "defaultBeta2": (0.2, False),
            "defaultEps": (0.2, False),
        }
    )
    optimizers.append([adam0, adam1])

    # Adaptive
    adaptive0 = popart.Adaptive(
        {
            "lossScaling": (10.0, False),
            "defaultLearningRate": (0.5, False),
            "defaultAlpha": (0.5, False),
            "defaultMomentum": (0.5, False),
            "defaultWeightDecay": (0.5, False),
            "defaultEps": (0.5, False),
        }
    )
    adaptive1 = popart.Adaptive(
        {
            "lossScaling": (0.2, False),
            "defaultLearningRate": (0.2, False),
            "defaultAlpha": (0.2, False),
            "defaultMomentum": (0.2, False),
            "defaultWeightDecay": (0.2, False),
            "defaultEps": (0.2, False),
        }
    )
    optimizers.append([adaptive0, adaptive1])

    return optimizers
