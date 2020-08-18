# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import numpy.random as npr
import popart
import onnx
import os
import pytest
import test_util as tu

batch_size = 12

# the number of classes, and the number of features at every depty of the net
hidden_size = 16

num_ipus = 2

# all input and weight values are initialized to 1 - U where U ~ Uniform[0,xi].
xi = 0.1

# learning rate
lr = 0.12

#weight decay
wd = 0.08

# beta 1 (Adam)
b1 = 0.9

# beta 2 (Adam)
b2 = 0.999

# setting this to, say, 100, and the test fails, see T24563
testTilesPerIPU = 1216

sgd_optimizer = popart.SGD({
    "defaultLearningRate": (lr, False),
    "defaultWeightDecay": (wd, False)
})

adam_optimizer = popart.Adam({
    "defaultLearningRate": (lr, False),
    "defaultBeta1": (b1, False),
    "defaultBeta2": (b2, False),
    "defaultWeightDecay": (wd, False),
    "defaultEps": (1e-6, True),
})

grad_accl_prefix = popart.reservedAcclPrefix()


def get_micro_batch_size(accum_factor):
    """
    no data replication, so micro batch size = batch size / accumlation factor
    """
    if (batch_size % accum_factor is not 0):
        raise RuntimeError("accum_factor is not a factor of batch_size")

    micro_batch_size = batch_size // accum_factor
    return micro_batch_size


def get_mm_model(accum_factor, enable_multi_ipu):
    """
    out = mm(mm(mm(mm(mm(mm(in)))))) all mm are hidden_size x hidden_size
    """

    np.random.seed(1234)

    micro_batch_size = get_micro_batch_size(accum_factor)

    builder = popart.Builder()
    input_shape = [micro_batch_size, hidden_size]
    input_tensor_name = builder.addInputTensor(
        popart.TensorInfo("FLOAT", input_shape))
    x = input_tensor_name

    # Put 50% on each IPU if we are using multi IPU
    for i in range(6):
        w = builder.addInitializedInputTensor(
            (1.0 - xi * npr.rand(hidden_size, hidden_size)).astype(np.float32),
            "weight")
        x = builder.aiOnnx.matmul([x, w])
        if enable_multi_ipu:
            if i > 3:
                builder.virtualGraph(x, 1)
            else:
                builder.virtualGraph(x, 0)

    label_tensor_name = builder.addInputTensor("INT32", [micro_batch_size])
    x = builder.aiGraphcore.nllloss([x, label_tensor_name],
                                    reduction=popart.ReductionType.Sum)
    if enable_multi_ipu:
        builder.virtualGraph(x, 1)

    initial_onnx_model = builder.getModelProto()

    return initial_onnx_model, input_tensor_name, x, label_tensor_name


def get_complex_model(accum_factor):
    """
    out = softmax(reshape(conv(exp(sin(x)), weights))
    """

    np.random.seed(1234)

    micro_batch_size = get_micro_batch_size(accum_factor)

    builder = popart.Builder()
    input_shape = [micro_batch_size, 2, 4, 4]
    input_tensor_name = builder.addInputTensor(
        popart.TensorInfo("FLOAT", input_shape))
    w0 = builder.addInitializedInputTensor(
        np.ones([2, 2, 3, 3]).astype(np.float32))
    x0 = input_tensor_name

    s0 = builder.aiOnnx.sin([x0], "s0")
    e0 = builder.aiOnnx.exp([s0], "e0")
    c0 = builder.aiOnnx.conv([e0, w0],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1],
                             debugPrefix="c0")

    r0 = builder.reshape_const(builder.aiOnnx, [c0], [micro_batch_size, 32])
    sm = builder.aiOnnx.softmax([r0], axis=1, debugPrefix="sfm")
    output_tensor_name = builder.aiGraphcore.identityloss(
        [sm], reduction=popart.ReductionType.Sum)

    builder.addOutputTensor(output_tensor_name)
    label_shape = [micro_batch_size]
    label_tensor_name = builder.addInputTensor(
        popart.TensorInfo("INT32", label_shape))

    initial_onnx_model = builder.getModelProto()
    return initial_onnx_model, input_tensor_name, output_tensor_name, label_tensor_name


def run_graph(optimizer, input_shape, initial_onnx_model, input_tensor_name,
              output_tensor_name, label_tensor_name, label_array, accum_factor,
              enable_accum, batches_per_step, number_of_steps,
              final_proto_filename, enable_multi_ipu, full_anchorage,
              inference_mode):

    art = popart.AnchorReturnType("All")
    anchorNames = {output_tensor_name: art}

    if full_anchorage:
        w0 = onnx.load_from_string(
            initial_onnx_model).graph.initializer[0].name

        anchorNames[popart.reservedGradientPrefix() + w0] = art

        if enable_accum:
            anchorNames[popart.reservedAcclPrefix() + w0] = art
            anchorNames[popart.reservedAcclToUpdatePrefix() + w0] = art

    opts = popart.SessionOptions()
    opts.enableGradientAccumulation = enable_accum
    opts.accumulationFactor = accum_factor
    opts.enableOutlining = False
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual if enable_multi_ipu else popart.VirtualGraphMode.Off

    if enable_multi_ipu:
        device = tu.create_test_device(numIpus=num_ipus,
                                       tilesPerIPU=testTilesPerIPU,
                                       opts={"compileIPUCode": False})
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    else:
        device = tu.create_test_device(tilesPerIPU=testTilesPerIPU,
                                       opts={"compileIPUCode": False})
        opts.virtualGraphMode = popart.VirtualGraphMode.Off

    # only for test purposes, inference with gradient_accumulation should never work
    if inference_mode:
        popart.InferenceSession(fnModel=initial_onnx_model,
                                dataFlow=popart.DataFlow(
                                    batches_per_step, anchorNames),
                                userOptions=opts,
                                deviceInfo=device)

    session = popart.TrainingSession(fnModel=initial_onnx_model,
                                     dataFlow=popart.DataFlow(
                                         batches_per_step, anchorNames),
                                     deviceInfo=device,
                                     loss=output_tensor_name,
                                     optimizer=optimizer,
                                     userOptions=opts)

    session.prepareDevice()
    session.weightsFromHost()

    anchor_arrays = session.initAnchorArrays()

    outer_dim = 1
    if batches_per_step > 1:
        outer_dim *= batches_per_step
        label_array = np.repeat(label_array[np.newaxis], batches_per_step, 0)
    if accum_factor > 1:
        outer_dim *= accum_factor
        label_array = label_array.reshape(
            [accum_factor * batches_per_step, -1])
    if outer_dim > 1:
        input_shape = [outer_dim] + input_shape

    stepio = popart.PyStepIO(
        {
            input_tensor_name:
            (1.0 - xi * npr.rand(*input_shape)).astype(np.float32),
            label_tensor_name:
            label_array.astype(np.int32)
        }, anchor_arrays)

    for i in range(number_of_steps):
        session.run(stepio)

    final_proto_file = "{}.onnx".format(final_proto_filename)
    session.modelToHost(final_proto_filename)

    return final_proto_filename, anchor_arrays


def run_complex_graph(optimizer, label_array, accum_factor, enable_accum,
                      batches_per_step, number_of_steps, final_proto_filename,
                      enable_multi_ipu, full_anchorage):

    if (enable_multi_ipu):
        raise RuntimeError("Cannot enable multi ipu in complex graph")

    initial_onnx_model, input_tensor_name, output_tensor_name, label_tensor_name = get_complex_model(
        accum_factor)

    final_proto_filename, anchor_arrays = run_graph(
        optimizer,
        input_shape=[get_micro_batch_size(accum_factor), 2, 4, 4],
        initial_onnx_model=initial_onnx_model,
        input_tensor_name=input_tensor_name,
        output_tensor_name=output_tensor_name,
        label_tensor_name=label_tensor_name,
        label_array=label_array,
        accum_factor=accum_factor,
        enable_accum=enable_accum,
        batches_per_step=batches_per_step,
        number_of_steps=number_of_steps,
        final_proto_filename=final_proto_filename,
        enable_multi_ipu=enable_multi_ipu,
        full_anchorage=full_anchorage,
        inference_mode=False)

    return initial_onnx_model, final_proto_filename, anchor_arrays


def run_mm_graph(optimizer,
                 label_array,
                 accum_factor,
                 enable_accum,
                 batches_per_step,
                 number_of_steps,
                 final_proto_filename,
                 enable_multi_ipu,
                 full_anchorage,
                 inference_mode=False):

    initial_onnx_model, input_tensor_name, output_tensor_name, label_tensor_name = get_mm_model(
        accum_factor, enable_multi_ipu)

    final_proto_filename, anchor_arrays = run_graph(
        optimizer,
        input_shape=[get_micro_batch_size(accum_factor), hidden_size],
        initial_onnx_model=initial_onnx_model,
        input_tensor_name=input_tensor_name,
        output_tensor_name=output_tensor_name,
        label_tensor_name=label_tensor_name,
        label_array=label_array,
        accum_factor=accum_factor,
        enable_accum=enable_accum,
        batches_per_step=batches_per_step,
        number_of_steps=number_of_steps,
        final_proto_filename=final_proto_filename,
        enable_multi_ipu=enable_multi_ipu,
        full_anchorage=full_anchorage,
        inference_mode=inference_mode)

    return initial_onnx_model, final_proto_filename, anchor_arrays


def check_models(model_init, modelA_fn, modelB_fn):
    """
    for each weight tensor, check the relative error. That is, 
    | model_accl - model_no_accl |_1 / | model_accl - model_initial|_1
    """
    modelA = onnx.load(modelA_fn)
    modelB = onnx.load(modelB_fn)

    #the initial model
    modelC = onnx.load_from_string(model_init)

    for w_i, weightA in enumerate(modelA.graph.initializer):
        # We need to avoid the gradient accl initializers as these won't be present
        # in the non grad accl models.
        if (popart.reservedAcclPrefix() not in weightA.name
                and popart.reservedAccl1Prefix() not in weightA.name
                and popart.reservedAccl2Prefix() not in weightA.name
                and popart.reservedStepPrefix() not in weightA.name
                and popart.reservedAccumPrefix() not in weightA.name):
            # where A, B, C are weight tensors,
            # |A - B|_1
            l1AB = 0
            # |B - C|_1
            l1BC = 0
            # |A - C|_1
            l1AC = 0
            for d_i, dataA in enumerate(weightA.float_data):
                dataB = modelB.graph.initializer[w_i].float_data[d_i]
                dataC = modelC.graph.initializer[w_i].float_data[d_i]

                # abs diff of 2 floats
                l1AB += np.abs(dataA - dataB)
                l1BC += np.abs(dataB - dataC)
                l1AC += np.abs(dataA - dataC)

            relative_error = l1AB / (l1AC)
            print(
                f"{weightA.name}: l1AB = %.2e,  l1AC = %.2e, l1BC = %.2e, relative error = %.2e"
                % (l1AB, l1AC, l1BC, relative_error))

            # check that the weights have moved enough for this to be a valid
            assert l1AC > 1e-3, "change since start of A = %.5f" % (l1AC, )
            assert l1BC > 1e-3, "change since start of B = %.5f" % (l1BC, )

            #relative error assertion
            assert 1e-5 > relative_error, "Relative error {}".format(
                relative_error)


@tu.requires_ipu_model
def test_gradient_accumulation_base(tmpdir):
    """
    base test (as simple as possible)
    """

    for graph_runner in [run_complex_graph, run_mm_graph]:

        np.random.seed(1234)
        label_array = np.random.randint(0, hidden_size, batch_size)

        accl_initial_proto, accl_proto_filename, accl_anchor_arrays = graph_runner(
            sgd_optimizer,
            label_array=label_array,
            accum_factor=4,
            enable_accum=True,
            batches_per_step=1,
            number_of_steps=1,
            final_proto_filename=os.path.join(tmpdir, "accl"),
            enable_multi_ipu=False,
            full_anchorage=False)

        no_accl_initial_proto, no_accl_proto_filename, no_accl_anchor_arrays = graph_runner(
            sgd_optimizer,
            label_array=label_array,
            accum_factor=1,
            enable_accum=False,
            batches_per_step=1,
            number_of_steps=1,
            final_proto_filename=os.path.join(tmpdir, "noAcc"),
            enable_multi_ipu=False,
            full_anchorage=False)

        check_models(accl_initial_proto, accl_proto_filename,
                     no_accl_proto_filename)


@tu.requires_ipu_model
def test_gradient_accumulation_multi_batch(tmpdir):
    """
    from _base: increase batches per step and number of steps
    """

    for graph_runner in [run_mm_graph, run_complex_graph]:
        np.random.seed(1234)
        label_array = np.random.randint(0, hidden_size, batch_size)

        accl_initial_proto, accl_proto_filename, accl_anchor_arrays = run_mm_graph(
            sgd_optimizer,
            label_array=label_array,
            accum_factor=4,
            enable_accum=True,
            batches_per_step=5,
            number_of_steps=3,
            final_proto_filename=os.path.join(tmpdir, "accl5batches3steps"),
            enable_multi_ipu=False,
            full_anchorage=False)

        no_accl_initial_proto, no_accl_proto_filename, no_accl_anchor_arrays = run_mm_graph(
            sgd_optimizer,
            label_array=label_array,
            accum_factor=1,
            enable_accum=False,
            batches_per_step=5,
            number_of_steps=3,
            final_proto_filename=os.path.join(tmpdir, "noAccl5batches3steps"),
            enable_multi_ipu=False,
            full_anchorage=False)

        check_models(accl_initial_proto, accl_proto_filename,
                     no_accl_proto_filename)


@tu.requires_ipu_model
def test_gradient_accumulation_multi_ipu(tmpdir):
    """
    from _multi_batch: enable multi ipus
    """
    np.random.seed(1234)
    label_array = np.random.randint(0, hidden_size, batch_size)

    accl_initial_proto, accl_proto_filename, accl_anchor_arrays = run_mm_graph(
        sgd_optimizer,
        label_array=label_array,
        accum_factor=4,
        enable_accum=True,
        batches_per_step=5,
        number_of_steps=3,
        final_proto_filename=os.path.join(tmpdir, "accl5batches3steps"),
        enable_multi_ipu=True,
        full_anchorage=False)

    no_accl_initial_proto, no_accl_proto_filename, no_accl_anchor_arrays = run_mm_graph(
        sgd_optimizer,
        label_array=label_array,
        accum_factor=1,
        enable_accum=False,
        batches_per_step=5,
        number_of_steps=3,
        final_proto_filename=os.path.join(tmpdir, "noAccl5batches3steps"),
        # we do not enable multiple IPUs in the baseline
        enable_multi_ipu=False,
        full_anchorage=False)

    check_models(accl_initial_proto, accl_proto_filename,
                 no_accl_proto_filename)


@tu.requires_ipu_model
def test_gradient_accumulation_error_inference(tmpdir):
    """
    confirm that there is an error if in inference mode
    """

    label_array = np.random.randint(0, hidden_size, batch_size)
    with pytest.raises(popart.popart_exception) as e_info:

        a, b, c = run_mm_graph(sgd_optimizer,
                               label_array=label_array,
                               accum_factor=4,
                               enable_accum=True,
                               batches_per_step=5,
                               number_of_steps=3,
                               final_proto_filename=os.path.join(
                                   tmpdir, "accl5batches3steps"),
                               enable_multi_ipu=True,
                               full_anchorage=False,
                               inference_mode=True)

    assert e_info.value.args[0].startswith(
        "Gradient Accumulation only available when training")


@tu.requires_ipu_model
def test_gradient_accumulation_error_accum_factor_invalid(tmpdir):
    """
    confirm that enable_accum = False => accum_factor = 1
    """
    label_array = np.random.randint(0, hidden_size, batch_size)
    with pytest.raises(popart.popart_exception) as e_info:

        a, b, c = run_mm_graph(sgd_optimizer,
                               label_array=label_array,
                               accum_factor=4,
                               enable_accum=False,
                               batches_per_step=5,
                               number_of_steps=3,
                               final_proto_filename=os.path.join(
                                   tmpdir, "accl5batches3steps"),
                               enable_multi_ipu=True,
                               full_anchorage=False,
                               inference_mode=False)

    assert e_info.value.args[0].startswith(
        "enableGradientAccumulation is false, but accumulationFactor > 1.")


@tu.requires_ipu_model
def test_gradient_accumulation_anchors(tmpdir):
    """
    Check that the accumulated gradients with gradient accumulation match
    the gradients without gradient accumulation enabled.
    """

    label_array = np.random.randint(0, hidden_size, batch_size)

    #TODO T11866 larger batches-per-step, first without weight decay, then with weight decay
    batches_per_step = 1

    accl_initial_proto, accl_proto_filename, accl_anchor_arrays = run_mm_graph(
        sgd_optimizer,
        label_array=label_array,
        accum_factor=4,
        enable_accum=True,
        batches_per_step=batches_per_step,
        number_of_steps=1,
        final_proto_filename=os.path.join(tmpdir,
                                          "accl5batches3stepsAnchorsTest"),
        enable_multi_ipu=False,
        full_anchorage=True,
        inference_mode=False)

    no_accl_initial_proto, no_accl_proto_filename, no_accl_anchor_arrays = run_mm_graph(
        sgd_optimizer,
        label_array=label_array,
        accum_factor=1,
        enable_accum=False,
        batches_per_step=batches_per_step,
        number_of_steps=1,
        final_proto_filename=os.path.join(tmpdir,
                                          "noAccl5batches3stepsAnchorsTest"),
        enable_multi_ipu=False,
        full_anchorage=True,
        inference_mode=False)

    w0_tensor = onnx.load_from_string(accl_initial_proto).graph.initializer[0]
    w0_name = w0_tensor.name

    full_batch_grad = no_accl_anchor_arrays[popart.reservedGradientPrefix() +
                                            w0_name]
    accl_grad = accl_anchor_arrays[popart.reservedAcclToUpdatePrefix() +
                                   w0_name]

    print("full batch grad shape is ")
    print(full_batch_grad.shape)

    print("accl grad shape is ")
    print(accl_grad.shape)

    if (batches_per_step > 1):
        #TODO T11866
        raise RuntimeError("batches per step > 1 needs investigation")

        for i in range(batches_per_step):
            print("\nbatch %d" % (i, ))
            print("Absolute accl grad  %.3f" % (np.sum(np.abs(accl_grad[i]))))
            print("Absolute no accl g  %.3f" %
                  (np.sum(np.abs(full_batch_grad[i]))))
            print("Absolute difference %.3f" %
                  (np.sum(np.abs(full_batch_grad[i] - accl_grad[i]))))

            print("Absolute difference %.3f" %
                  (np.sum(np.abs(full_batch_grad[i] - adjusted_accl_grad[i]))))

    else:
        accl_grad_abs_sum = np.sum(np.abs(accl_grad))
        print("Absolute accl grad  %.3f" % (accl_grad_abs_sum))

        # initialising as per equations. When velocity scaling != 1 this may need changing T12001
        adjusted_accl_grad = accl_grad[-1].flatten().copy()
        for i, v in enumerate(w0_tensor.float_data):
            adjusted_accl_grad[i] -= wd * v

        adjusted_accl_grad_abs_sum = np.sum(np.abs(adjusted_accl_grad))
        print("Absolute accl grad, adjusted for weight decay %.3f" %
              (adjusted_accl_grad_abs_sum))

        full_batch_abs_sum = np.sum(np.abs(full_batch_grad))
        print("Absolute no accl g  %.3f" % (full_batch_abs_sum))

        abs_diff = np.sum(
            np.abs(full_batch_grad.flatten() - adjusted_accl_grad))
        print("Absolute difference %.3f" % (abs_diff))

        assert (abs_diff / (full_batch_abs_sum + accl_grad_abs_sum) < 1e-5)


@tu.requires_ipu_model
def test_gradient_accumulation_model_proto(tmpdir):
    np.random.seed(1234)
    label_array = np.random.randint(0, hidden_size, batch_size)
    accl_initial_proto, accl_proto_filename, accl_anchor_arrays = run_mm_graph(
        sgd_optimizer,
        label_array=label_array,
        accum_factor=4,
        enable_accum=True,
        batches_per_step=5,
        number_of_steps=3,
        final_proto_filename=os.path.join(tmpdir, "accl5batches3steps"),
        enable_multi_ipu=False,
        full_anchorage=False)

    model = onnx.load(accl_proto_filename)
    names = [t.name for t in model.graph.initializer]

    grad_accl_names = []
    weight_names = []
    for name in names:
        if grad_accl_prefix in name:
            grad_accl_names.append(name)
        elif "weight" in name:
            weight_names.append(name)

    # Model should have 6 weight tensors
    assert len(weight_names) == 6
    assert len(grad_accl_names) == len(weight_names)

    tensor_mapping = {}
    for tensor in model.graph.initializer:
        tensor_mapping[tensor.name] = tensor

    rev_map = {}
    for w_name in weight_names:
        assert grad_accl_prefix + w_name in grad_accl_names
        rev_map[grad_accl_prefix + w_name] = w_name

    for g_a_name in grad_accl_names:
        weight_tensor = tensor_mapping[rev_map[g_a_name]]
        g_a_tensor = tensor_mapping[g_a_name]
        for d_i, v in enumerate(weight_tensor.float_data):
            # initialisation as per equations. When velocity scaling != 1 this
            # will need changing : T12001
            assert g_a_tensor.float_data[d_i] - v * wd < 1e-8


def test_loading_saved_gradient_accumulationt_tesors(tmpdir):
    """
    1. Build a model with matmuls, no grad accumulation
    2. Write out onnx model, verify initializers contain no accl tensors
    3. Create session with model, verify accl tensors initialised correctly
    4. Do session.run(), write out model, verify accl tensors have been updated
    5. Create new session with same model. This time before run, write out model
       and check tensors are still there, with the same value
    """

    # 1.
    accum_factor = 4
    [onnx_model, input_name, output_name,
     lb_name] = get_mm_model(accum_factor=accum_factor, enable_multi_ipu=False)

    # 2.
    model = onnx.load_from_string(onnx_model)
    names = [t.name for t in model.graph.initializer]
    for name in names:
        assert grad_accl_prefix not in name

    def getTrainingSession(fn):
        opts = popart.SessionOptions()
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = accum_factor
        opts.disableGradAccumulationTensorStreams = False
        sess = popart.TrainingSession(
            fnModel=fn,
            dataFlow=popart.DataFlow(1, {}),
            deviceInfo=tu.create_test_device(tilesPerIPU=testTilesPerIPU),
            loss=output_name,
            optimizer=sgd_optimizer,
            userOptions=opts)
        sess.prepareDevice()

        sess.weightsFromHost()
        return sess

    # 3.
    sess = getTrainingSession(onnx_model)
    fn = os.path.join(tmpdir, "withInitZeroAcclTensors.onnx")
    sess.modelToHost(fn)
    model = onnx.load(fn)
    weights = {}
    accls = {}
    for t in model.graph.initializer:
        if grad_accl_prefix in t.name:
            accls[t.name] = t.float_data
        else:
            weights[t.name] = t.float_data
    for name in weights:
        t_weight = np.asarray(weights[name])
        t_accl = np.asarray(accls[grad_accl_prefix + name])

    # 4.
    input_shape = [accum_factor] + sess.getInfo(input_name).shape()
    stepio = popart.PyStepIO(
        {
            input_name: npr.rand(*input_shape).astype(np.float32),
            lb_name: np.ones(batch_size).astype(np.int32),
        }, sess.initAnchorArrays())
    sess.run(stepio)
    fn = os.path.join(tmpdir, "withUpdatedAcclTensors.onnx")
    sess.modelToHost(fn)
    model = onnx.load(fn)
    up_accls = {}
    for t in model.graph.initializer:
        if grad_accl_prefix in t.name:
            up_accls[t.name] = np.asarray(t.float_data)
            assert np.allclose(np.asarray(t.float_data),
                               np.asarray(accls[t.name])) is False

    # 5.
    sess = getTrainingSession(fn)
    fn = os.path.join(tmpdir, "withUpdatedAcclTensors_check.onnx")
    sess.modelToHost(fn)
    model = onnx.load(fn)
    for t in model.graph.initializer:
        if grad_accl_prefix in t.name:
            assert np.array_equal(up_accls[t.name], np.asarray(t.float_data))


@tu.requires_ipu_model
def test_adam_gradient_accumulation_base(tmpdir):
    """
    base test (as simple as possible)
    """

    for graph_runner in [run_complex_graph, run_mm_graph]:

        np.random.seed(1234)
        label_array = np.random.randint(0, hidden_size, batch_size)

        accum_initial_proto, accum_proto_filename, accum_anchor_arrays = graph_runner(
            adam_optimizer,
            label_array=label_array,
            accum_factor=4,
            enable_accum=True,
            batches_per_step=1,
            number_of_steps=1,
            final_proto_filename=os.path.join(tmpdir, "adamAccum"),
            enable_multi_ipu=False,
            full_anchorage=False)

        no_accum_initial_proto, no_accum_proto_filename, no_accum_anchor_arrays = graph_runner(
            adam_optimizer,
            label_array=label_array,
            accum_factor=1,
            enable_accum=False,
            batches_per_step=1,
            number_of_steps=1,
            final_proto_filename=os.path.join(tmpdir, "adamNoAccum"),
            enable_multi_ipu=False,
            full_anchorage=False)

        check_models(accum_initial_proto, accum_proto_filename,
                     no_accum_proto_filename)


@tu.requires_ipu_model
def test_adam_gradient_accumulation_multi_batch(tmpdir):
    """
    from _base: increase batches per step and number of steps
    """

    for graph_runner in [run_mm_graph, run_complex_graph]:
        np.random.seed(1234)
        label_array = np.random.randint(0, hidden_size, batch_size)

        accum_initial_proto, accum_proto_filename, accum_anchor_arrays = run_mm_graph(
            adam_optimizer,
            label_array=label_array,
            accum_factor=4,
            enable_accum=True,
            batches_per_step=5,
            number_of_steps=3,
            final_proto_filename=os.path.join(tmpdir,
                                              "adamAccum5batches3steps"),
            enable_multi_ipu=False,
            full_anchorage=False)

        no_accum_initial_proto, no_accum_proto_filename, no_accum_anchor_arrays = run_mm_graph(
            adam_optimizer,
            label_array=label_array,
            accum_factor=1,
            enable_accum=False,
            batches_per_step=5,
            number_of_steps=3,
            final_proto_filename=os.path.join(tmpdir,
                                              "adamNoAccum5batches3steps"),
            enable_multi_ipu=False,
            full_anchorage=False)

        check_models(accum_initial_proto, accum_proto_filename,
                     no_accum_proto_filename)


@tu.requires_ipu_model
def test_adam_gradient_accumulation_multi_ipu(tmpdir):
    """
    from _multi_batch: enable multi ipus
    """
    np.random.seed(1234)
    label_array = np.random.randint(0, hidden_size, batch_size)

    accum_initial_proto, accum_proto_filename, accum_anchor_arrays = run_mm_graph(
        adam_optimizer,
        label_array=label_array,
        accum_factor=4,
        enable_accum=True,
        batches_per_step=5,
        number_of_steps=3,
        final_proto_filename=os.path.join(tmpdir, "adamAccum5batches3steps"),
        enable_multi_ipu=True,
        full_anchorage=False)

    no_accum_initial_proto, no_accum_proto_filename, no_accum_anchor_arrays = run_mm_graph(
        adam_optimizer,
        label_array=label_array,
        accum_factor=1,
        enable_accum=False,
        batches_per_step=5,
        number_of_steps=3,
        final_proto_filename=os.path.join(tmpdir, "adamNoAccum5batches3steps"),
        # we do not enable multiple IPUs in the baseline
        enable_multi_ipu=False,
        full_anchorage=False)

    check_models(accum_initial_proto, accum_proto_filename,
                 no_accum_proto_filename)


@tu.requires_ipu_model
def test_adam_gradient_accumulation_model_proto(tmpdir):
    batches_per_step = 5
    for steps in [0, 3]:
        np.random.seed(1234)
        label_array = np.random.randint(0, hidden_size, batch_size)
        accl_initial_proto, accl_proto_filename, accl_anchor_arrays = run_mm_graph(
            adam_optimizer,
            label_array=label_array,
            accum_factor=4,
            enable_accum=True,
            batches_per_step=batches_per_step,
            number_of_steps=steps,
            final_proto_filename=os.path.join(tmpdir, "accl5batches3steps"),
            enable_multi_ipu=False,
            full_anchorage=False)

        model = onnx.load(accl_proto_filename)
        names = [t.name for t in model.graph.initializer]

        weight_names = []
        accum_names = []
        accl1_names = []
        accl2_names = []
        step_names = []

        for name in names:
            if popart.reservedAccumPrefix() in name:
                accum_names.append(name)
            elif popart.reservedAccl1Prefix() in name:
                accl1_names.append(name)
            elif popart.reservedAccl2Prefix() in name:
                accl2_names.append(name)
            elif popart.reservedStepPrefix() in name:
                step_names.append(name)
            elif "weight" in name:
                weight_names.append(name)

        # Model should have 6 weight tensors
        assert len(weight_names) == 6
        assert len(accum_names) == len(weight_names)
        assert len(accl1_names) == len(weight_names)
        assert len(accl2_names) == len(weight_names)
        assert len(step_names) == len(weight_names)

        tensor_mapping = {}
        for tensor in model.graph.initializer:
            tensor_mapping[tensor.name] = tensor

        for w_name in weight_names:
            assert popart.reservedAccumPrefix() + w_name in accum_names
            assert popart.reservedAccl1Prefix() + w_name in accl1_names
            assert popart.reservedAccl2Prefix() + w_name in accl2_names
            assert popart.reservedStepPrefix() + w_name in step_names

        if steps == 0:
            for name in accum_names + accl1_names + accl2_names + step_names:
                tensor = tensor_mapping[name]
                # All Adam states are initialized to zero
                assert np.allclose(tensor.float_data, 0.0)
        else:
            for name in step_names:
                tensor = tensor_mapping[name]
                # Steps counted correctly
                assert tensor.float_data[0] == steps * batches_per_step


def test_adam_loading_saved_gradient_accumulationt_tesors(tmpdir):
    """
    1. Build a model with matmuls, no grad accumulation
    2. Write out onnx model, verify initializers contain no accum tensors
    3. Create session with model, verify accl tensors initialised correctly
    4. Do session.run(), write out model, verify accl tensors have been updated
    5. Create new session with same model. This time before run, write out model
       and check tensors are still there, with the same value
    """

    # 1.
    accum_factor = 4
    [onnx_model, input_name, output_name,
     lb_name] = get_mm_model(accum_factor=accum_factor, enable_multi_ipu=False)

    # 2.
    model = onnx.load_from_string(onnx_model)
    names = [t.name for t in model.graph.initializer]
    for name in names:
        assert popart.reservedAccumPrefix() not in name

    def getTrainingSession(fn):
        opts = popart.SessionOptions()
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = accum_factor
        opts.disableGradAccumulationTensorStreams = False
        sess = popart.TrainingSession(
            fnModel=fn,
            dataFlow=popart.DataFlow(1, {}),
            deviceInfo=tu.create_test_device(tilesPerIPU=testTilesPerIPU),
            loss=output_name,
            optimizer=adam_optimizer,
            userOptions=opts)
        sess.prepareDevice()

        sess.weightsFromHost()
        return sess

    # 3.
    sess = getTrainingSession(onnx_model)
    fn = os.path.join(tmpdir, "withInitZeroAccumTensors.onnx")
    sess.modelToHost(fn)
    model = onnx.load(fn)
    weights = {}
    optstates = {}
    for t in model.graph.initializer:
        if (popart.reservedAccumPrefix() in t.name
                or popart.reservedAccl1Prefix() in t.name
                or popart.reservedAccl2Prefix() in t.name
                or popart.reservedStepPrefix() in t.name):
            optstates[t.name] = t.float_data
            assert np.allclose(np.asarray(t.float_data), 0.0)
        else:
            weights[t.name] = t.float_data

    # 4.
    input_shape = [accum_factor] + sess.getInfo(input_name).shape()
    stepio = popart.PyStepIO(
        {
            input_name: npr.rand(*input_shape).astype(np.float32),
            lb_name: np.ones(batch_size).astype(np.int32),
        }, sess.initAnchorArrays())
    sess.run(stepio)
    fn = os.path.join(tmpdir, "withUpdatedAcclTensors.onnx")
    sess.modelToHost(fn)
    model = onnx.load(fn)
    for t in model.graph.initializer:
        if (popart.reservedAccl1Prefix() in t.name
                or popart.reservedAccl2Prefix() in t.name
                or popart.reservedStepPrefix() in t.name):
            # Nonzero, updated accl1, accl2 and step tensors
            assert np.allclose(np.asarray(t.float_data),
                               optstates[t.name]) is False
            optstates[t.name] = np.asarray(t.float_data)
        elif popart.reservedAccumPrefix() in t.name:
            # Because the accumulator is always set to zero after being applied
            # to accl1 and accl2
            assert np.allclose(np.asarray(t.float_data), 0.0)
            optstates[t.name] = np.asarray(t.float_data)

    # 5.
    sess = getTrainingSession(fn)
    fn = os.path.join(tmpdir, "withUpdatedAcclTensors_check.onnx")
    sess.modelToHost(fn)
    model = onnx.load(fn)
    for t in model.graph.initializer:
        if (popart.reservedAccumPrefix() in t.name
                or popart.reservedAccl1Prefix() in t.name
                or popart.reservedAccl2Prefix() in t.name
                or popart.reservedStepPrefix() in t.name):
            assert np.array_equal(optstates[t.name], np.asarray(t.float_data))
