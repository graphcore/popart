# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import onnx
from onnx import numpy_helper
import pytest
import test_util as tu
import torch


def loss_scaling_test(constLossScaling):

    # In previous implementations of popart loss scaling only worked for
    # nll/l1 losses. This test is to ensure it works for any loss.
    #
    # For a computation matmul(c1,p1) where c1 is a constant matrix and
    # p1 is a parameter, calculate loss as reducesum(matmul(c1,p1)) and
    # update p1 using a optimizer that 'uses' loss scaling. Do this
    # for both pytorch and popart and compare the result.

    np.random.seed(1)
    sgd_learning_rate = 0.1
    sgd_moment = 0.9
    c1_shape = (1, 1)
    p1_shape = (1, 1)
    c1_init = np.random.rand(*c1_shape).astype(np.float32)
    p1_init = np.random.rand(*p1_shape).astype(np.float32)
    out_init = np.matmul(c1_init, p1_init)

    def get_updated_p1_popart():
        builder = popart.Builder()

        # Computation is out = matmul(i1, i2)
        c1 = builder.addInputTensor(popart.TensorInfo("FLOAT", c1_shape))
        p1 = builder.addInitializedInputTensor(p1_init)
        out = builder.aiOnnx.matmul([c1, p1])

        # Set up a training session.
        device = tu.create_test_device()
        dataFlow = popart.DataFlow(
            1, {
                c1: popart.AnchorReturnType("Final"),
                p1: popart.AnchorReturnType("Final"),
                out: popart.AnchorReturnType("Final")
            })

        # We're testing losses other than nll/l1 work.
        loss = builder.aiOnnx.reducesum([out])
        optimizer = popart.SGD({
            "defaultLearningRate": (sgd_learning_rate, True),
            "defaultMomentum": (sgd_moment, False),
            "lossScaling": (200, constLossScaling)
        })
        session = popart.TrainingSession(builder.getModelProto(),
                                         deviceInfo=device,
                                         dataFlow=dataFlow,
                                         loss=loss,
                                         optimizer=optimizer)

        session.prepareDevice()
        session.weightsFromHost()

        # Run the popart session to get an answer.
        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({c1: c1_init}, anchors)
        session.run(stepio)
        return anchors[c1], anchors[p1], anchors[out]

    def get_updated_p1_pytorch():

        # Computation is out = matmul(i1, i2)
        c1 = torch.tensor(c1_init, requires_grad=False)
        p1 = torch.tensor(p1_init, requires_grad=True)
        out = torch.matmul(c1, p1)

        # Set up optimizer, compute loss.
        optimizer = torch.optim.SGD([p1],
                                    lr=sgd_learning_rate,
                                    momentum=sgd_moment)
        optimizer.zero_grad()
        loss = torch.sum(out)

        # Compute gradient and optimize.
        loss.backward()
        optimizer.step()

        # Return the tensors.
        return c1.detach().numpy(), p1.detach().numpy(), out.detach().numpy()

    c1_popart, p1_popart, out_popart = get_updated_p1_popart()
    c1_pytorch, p1_pytorch, out_pytorch = get_updated_p1_pytorch()

    # We're not expecting changes in c1 or out, check anyway.
    assert (np.isclose(c1_popart, c1_init))
    assert (np.isclose(c1_pytorch, c1_init))
    assert (np.isclose(out_popart, out_init))
    assert (np.isclose(out_pytorch, out_init))

    # We expect p1 to match.
    assert (np.isclose(
        p1_popart, p1_pytorch
    )), f'Expected p1_popart={p1_popart} to match p1_pytorch={p1_pytorch}'


def test_loss_scaling_with_const():
    loss_scaling_test(True)


def test_loss_scaling_with_nonconst():
    loss_scaling_test(False)


def test_auto_loss_scaling_with_inference_session():
    """
    Create an InferenceSession with auto loss scaling enabled. Observe an
    error from the auto loss scale transform
    """
    builder = popart.Builder()

    t0 = builder.addInputTensor("FLOAT", [2, 2])
    out = builder.aiOnnx.matmul([t0, t0])

    opts = popart.SessionOptions()
    opts.enableAutomaticLossScaling = True

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(builder.getModelProto(),
                                          deviceInfo=tu.create_test_device(),
                                          dataFlow=popart.DataFlow(1, [out]),
                                          userOptions=opts)
    assert e_info.value.args[0].endswith("Only compatible when doing training")


def test_auto_loss_scaling_with_const_loss_scale_tensor():
    """
    Create a session with auto loss scaling enabled, and with an optimizer
    with a constant loss scale value. Observe an error from the auto loss
    scale transform
    """
    builder = popart.Builder()

    t0 = builder.addInputTensor("FLOAT", [2, 2])
    t1_data = np.random.rand(2, 2).astype(np.float32)
    t1 = builder.addInitializedInputTensor(t1_data)
    out = builder.aiOnnx.matmul([t0, t1])
    loss = builder.aiGraphcore.identityloss([out])

    makeLossScalingTensorConst = True
    optimizer = popart.SGD({"lossScaling": (2, makeLossScalingTensorConst)})

    opts = popart.SessionOptions()
    opts.enableAutomaticLossScaling = True

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(builder.getModelProto(),
                                         deviceInfo=tu.create_test_device(),
                                         dataFlow=popart.DataFlow(1, []),
                                         loss=loss,
                                         optimizer=optimizer,
                                         userOptions=opts)
    assert e_info.value.args[0].endswith("The optimizer must have non-const loss scaling")


def test_auto_loss_scaling_with_no_tracked_tensors():
    """
    Build a model with ops, the outputs of which the auto loss scale transform
    does not decide to 'track'. Observe an error from the auto loss scale
    transform
    """
    builder = popart.Builder()

    t0 = builder.addInputTensor("FLOAT", [2, 2])
    out = builder.aiOnnx.relu([t0])
    loss = builder.aiGraphcore.identityloss([out])

    optimizer = popart.SGD({"lossScaling": (2, False)})

    opts = popart.SessionOptions()
    opts.enableAutomaticLossScaling = True

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(builder.getModelProto(),
                                         deviceInfo=tu.create_test_device(),
                                         dataFlow=popart.DataFlow(1, [loss]),
                                         loss=loss,
                                         optimizer=optimizer,
                                         userOptions=opts)
    assert e_info.value.args[0].endswith("No tracked tensors were found")


def getModelProto(shard=False, pipeline=False):
    """
    Create a simple model:

    in0 -- Matmul -- t0 -- Relu -- t1 -- Conv -- t1 -- NllLoss -- loss
    w0  ---'                              /             /
    w1  ---------------------------------       label -

    whose graph after auto-grad will contain tensors whose statistics the
    automatic loss scale transform will track to adjust the loss scale factor
    """
    builder = popart.Builder()

    t_shape = [2, 1, 2, 2]
    t0 = builder.addInputTensor("FLOAT16", t_shape)
    t_data = np.random.rand(*t_shape).astype(np.float16)
    t1 = builder.addInitializedInputTensor(t_data)
    t2 = builder.addInitializedInputTensor(t_data)    
    mm = builder.aiOnnx.matmul([t0, t1])
    r = builder.aiOnnx.relu([mm])
    conv = builder.aiOnnx.conv([r, t2])
    rs = builder.reshape_const(builder.aiOnnx, [conv], [2, 2])
    sf = builder.aiOnnx.softmax([rs])
    label_shape = [2]
    labels = builder.addInputTensor("INT32", label_shape)
    loss = builder.aiGraphcore.nllloss([sf, labels])

    if shard:
        builder.virtualGraph(mm, 0)
        builder.virtualGraph(r, 0)
        builder.virtualGraph(conv, 1)
        builder.virtualGraph(rs, 1)
        builder.virtualGraph(sf, 1)
        builder.virtualGraph(loss, 1)
    if pipeline:
        builder.pipelineStage(mm, 0)
        builder.pipelineStage(r, 0)
        builder.pipelineStage(conv, 1)
        builder.pipelineStage(rs, 1)
        builder.pipelineStage(sf, 1)
        builder.pipelineStage(loss, 1)

    return loss, builder.getModelProto(), t0, t_shape, labels, label_shape


def test_auto_loss_scaling_expected_loss_scale_tensor_values():
    """
    Pick a very small loss scale value so that gradient tensor values occupy
    the lower part of the fp16 dynamic range.
    Observe that this is the case, and that the loss scale factor is adjusted
    as expected after each weight update.
    """
    init_loss_scale = np.finfo(np.float16).eps * 2
    optimizer = popart.SGD({"lossScaling": (init_loss_scale, False)})

    opts = popart.SessionOptions()
    opts.enableAutomaticLossScaling = True

    loss, proto, t0, t_shape, label, label_shape = getModelProto()
    bps = 4

    loss_scale_id = "lossScaling_FLOAT16_updated"
    gradient_anchor_ids = ["Gradient___init_input/1", "Gradient___init_input"]
    anchor_ids = gradient_anchor_ids + [loss_scale_id]

    session = popart.TrainingSession(fnModel=proto,
                                     deviceInfo=tu.create_test_device(),
                                     dataFlow=popart.DataFlow(bps, anchor_ids),
                                     loss=loss,
                                     optimizer=optimizer,
                                     userOptions=opts)
    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()

    t0_data = np.random.rand(bps, *t_shape).astype(np.float16)
    label_data = np.random.randint(0, label_shape[0], bps*label_shape[0]).astype(np.int32)
    stepio = popart.PyStepIO({t0: t0_data, label: label_data}, anchors)
    session.run(stepio)

    # Manually determine the direction of the loss scale update
    f16_max = np.finfo(np.float16).max
    histogram_bin_edges = [-1, 0.5 * f16_max, f16_max]
    prev_loss_scale = init_loss_scale
    for i in range(bps):
        num_small_grad_elements = 0
        num_large_grad_elements = 0
        print(anchors[loss_scale_id][i])
        for id in gradient_anchor_ids:
            abs_grad_data = np.abs(anchors[id][i])
            hist, _ = np.histogram(abs_grad_data, histogram_bin_edges)
            num_small_grad_elements += hist[0]
            num_large_grad_elements += hist[1]

        proportion_small_grad_elements = num_small_grad_elements / (num_small_grad_elements + num_large_grad_elements)
        # Observe that the proportion of small grad elements is large, as we
        # have started with a very small loss scale
        assert proportion_small_grad_elements > 0.9

        # Therefore the loss scale will increase for each batch in the step
        if i > 0:
            assert anchors[loss_scale_id][i] > prev_loss_scale
            prev_loss_scale = anchors[loss_scale_id][i]


def test_auto_loss_scaling_and_grad_accumulation():
    """
    Create a Session with automatic loss scaling and gradient accumulation
    enabled, and see that an incompatibility error is thrown.
    """
    builder = popart.Builder()

    t0 = builder.addInputTensor("FLOAT", [2, 2])
    mm0 = builder.aiOnnx.matmul([t0, t0])
    loss = builder.aiGraphcore.identityloss([mm0])

    optimizer = popart.SGD({"lossScaling": (2, False)})

    opts = popart.SessionOptions()
    opts.enableAutomaticLossScaling = True
    opts.enableGradientAccumulation = True

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(builder.getModelProto(),
                                         deviceInfo=tu.create_test_device(),
                                         dataFlow=popart.DataFlow(1, [loss]),
                                         loss=loss,
                                         optimizer=optimizer,
                                         userOptions=opts)
    assert e_info.value.args[0].endswith("Automatic loss scaling is not currently supported when the 'enableGradientAccumulation' SessionOption is set to 'true'")


def test_auto_loss_scaling_with_non_sgd_optimizer():
    """
    Create a Session with automatic loss scaling and a non-sgd optimizer,
    and see that an incompatibility error is thrown.
    """
    builder = popart.Builder()

    t0 = builder.addInputTensor("FLOAT", [2, 2])
    mm0 = builder.aiOnnx.matmul([t0, t0])
    loss = builder.aiGraphcore.identityloss([mm0])

    optimizer = popart.Adam({"lossScaling": (2, False)})

    opts = popart.SessionOptions()
    opts.enableAutomaticLossScaling = True

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(builder.getModelProto(),
                                         deviceInfo=tu.create_test_device(),
                                         dataFlow=popart.DataFlow(1, [loss]),
                                         loss=loss,
                                         optimizer=optimizer,
                                         userOptions=opts)
    assert e_info.value.args[0].endswith("Only compatible when using the SGD optimizer type, but you are using 'Adam'")


def test_auto_loss_scaling_sgd_with_specific_optimizer_values():
    """
    Create a Session with automatic loss scaling and an optimizer with a
    weight-specific optimizer value, and see that an incompatibility error is
    thrown.
    """
    builder = popart.Builder()
    t0 = builder.addInputTensor("FLOAT", [2, 2])
    t1_data = np.random.rand(2, 2).astype(np.float32)
    t1 = builder.addInitializedInputTensor(t1_data)
    mm0 = builder.aiOnnx.matmul([t0, t1])
    loss = builder.aiGraphcore.identityloss([mm0])

    optimizer = popart.SGD({"lossScaling": (2, False),
                            "defaultLearningRate": (0.2, False)})
    optimizer.insertSpecific(t1, {"learningRate": (0.1, False)})

    opts = popart.SessionOptions()
    opts.enableAutomaticLossScaling = True

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(builder.getModelProto(),
                                         deviceInfo=tu.create_test_device(),
                                         dataFlow=popart.DataFlow(1, [loss]),
                                         loss=loss,
                                         optimizer=optimizer,
                                         userOptions=opts)
    assert e_info.value.args[0].endswith("Not compatible with weight-specific optimizer values")


def test_auto_loss_scaling_with_mixed_precision_trackable_tensors():
    """
    Create a Session with automatic loss scaling and a model that contains
    both fp32 and fp16 initializers, and see that no incompatibility error is
    thrown.
    """
    builder = popart.Builder()
    t0 = builder.addInputTensor("FLOAT", [2, 2])
    t1_data = np.random.rand(2, 2).astype(np.float32)
    t1 = builder.addInitializedInputTensor(t1_data)
    mm0 = builder.aiOnnx.matmul([t0, t1])
    t2 = builder.aiOnnx.cast([mm0], "FLOAT16")
    t3 = builder.addInputTensor("FLOAT16", [2, 2])
    mm1 = builder.aiOnnx.matmul([t2, t3])
    loss = builder.aiGraphcore.identityloss([mm1])

    optimizer = popart.SGD({"lossScaling": (2, False)})

    opts = popart.SessionOptions()
    opts.enableAutomaticLossScaling = True

    session = popart.TrainingSession(builder.getModelProto(),
                                     deviceInfo=tu.create_test_device(),
                                     dataFlow=popart.DataFlow(1, [loss]),
                                     loss=loss,
                                     optimizer=optimizer,
                                     userOptions=opts)
    session.prepareDevice()

def compare_weights(session0, session1, tmpdir):
    ref_path = str(tmpdir / f"ref_session.onnx")
    session0.modelToHost(ref_path)
    session0_proto = onnx.load(ref_path)
    session0_weights = {}
    session1_weights = {}
    for i in range(len(session0_proto.graph.initializer)):
        init = session0_proto.graph.initializer[i]
        session0_weights[init.name] = np.empty(shape=init.dims, dtype=np.float16)
        session1_weights[init.name] = np.empty(shape=init.dims, dtype=np.float16)

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


def run_automatic_loss_scaling_comparison_test(tmpdir, shard=False, pipeline=False, replicate=False):
    """
    An integration test: verify that the weight updats computed by a session
    with auto loss scaling (ALS) enabled are identical to those with ALS
    disabled.
    """
    loss, proto, t0, t_shape, label, label_shape = getModelProto(shard=shard, pipeline=pipeline)
    bps = 4
    step_size = bps
    if replicate:
        replicas = 2
        step_size *= replicas
    init_ls = 10.0
    optimizer = popart.SGD({"lossScaling": (init_ls, False),
                            "defaultMomentum": (0.5, False),
                            "defaultVelocityScaling": (0.5, False),
                            "defaultDampening": (0.5, False),
                            "defaultWeightDecay": (0.5, False)})

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

    ref_session = popart.TrainingSession(fnModel=proto,
                                         deviceInfo=tu.create_test_device(num_ipus),
                                         dataFlow=popart.DataFlow(bps, []),
                                         loss=loss,
                                         optimizer=optimizer,
                                         userOptions=opts)
    ref_session.prepareDevice()
    ref_session.weightsFromHost()
    ref_anchors = ref_session.initAnchorArrays()


    opts.enableAutomaticLossScaling = True

    ls_id = "lossScaling_FLOAT16_updated"
    als_session = popart.TrainingSession(fnModel=proto,
                                         deviceInfo=tu.create_test_device(num_ipus),
                                         dataFlow=popart.DataFlow(bps, [ls_id]),
                                         loss=loss,
                                         optimizer=optimizer,
                                         userOptions=opts)
    als_session.prepareDevice()
    als_session.weightsFromHost()
    als_anchors = als_session.initAnchorArrays()

    t0_data = np.random.rand(step_size, *t_shape).astype(np.float16)
    label_data = np.random.randint(0, label_shape[0], step_size*label_shape[0]).astype(np.int32)
    inputs = {t0: t0_data, label: label_data}

    # Run once
    ref_session.run(popart.PyStepIO(inputs, ref_anchors))
    als_session.run(popart.PyStepIO(inputs, als_anchors))

    # Verify that the loss scale has changed from its initial value
    for i in range(bps):
        for ls in als_anchors[ls_id].flatten():
            assert ls != init_ls

    # Update the optimizer
    new_optimizer = popart.SGD({"lossScaling": (0.2, False),
                                "defaultMomentum": (0.2, False),
                                "defaultVelocityScaling": (0.2, False),
                                "defaultDampening": (0.2, False),
                                "defaultWeightDecay": (0.2, False)})
    ref_session.updateOptimizerFromHost(new_optimizer)
    als_session.updateOptimizerFromHost(new_optimizer)

    # Run a second time
    ref_session.run(popart.PyStepIO(inputs, ref_anchors))
    als_session.run(popart.PyStepIO(inputs, als_anchors))

    # Verify that the weight updates are identitcal for a.l.s vs a reference
    # session
    compare_weights(ref_session, als_session, tmpdir)


def test_auto_loss_scaling_identical_weight_updates(tmpdir):
    run_automatic_loss_scaling_comparison_test(tmpdir)


@tu.requires_ipu_model
def test_auto_loss_scaling_identical_weight_updates_sharded(tmpdir):
    run_automatic_loss_scaling_comparison_test(tmpdir, shard=True)


@pytest.mark.skip("T33956: ALS not supported with pipelined models")
@tu.requires_ipu_model
def test_auto_loss_scaling_identical_weight_updates_pipelined(tmpdir):
    run_automatic_loss_scaling_comparison_test(tmpdir, shard=True, pipeline=True)
