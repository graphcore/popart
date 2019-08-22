import numpy as np
import popart
import onnx
import pytest
import test_util as tu

batch_size = 16
hidden_size = 16
num_ipus = 2


def run_graph(labelArray,
              accl_steps,
              enable_accl,
              batches_per_step,
              proto_name,
              enable_multi_ipu,
              return_stepio=False):
    micro_batch_size = batch_size // accl_steps

    builder = popart.Builder()

    input_shape = [micro_batch_size, hidden_size]
    input_ = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape))

    x = input_
    # Put 50% on each IPU if we are using multi IPU
    with builder.virtualGraph(0):
        for i in range(3):
            w = builder.addInitializedInputTensor(
                np.ones([hidden_size, hidden_size]).astype(np.float32))
            if i == 0: w0 = w
            x = builder.aiOnnx.matmul([x, w])
    with builder.virtualGraph(1 if enable_multi_ipu else 0):
        for i in range(3):
            w = builder.addInitializedInputTensor(
                np.ones([hidden_size, hidden_size]).astype(np.float32))
            x = builder.aiOnnx.matmul([x, w])
    output = x

    builder.addOutputTensor(output)

    label_shape = [micro_batch_size]
    label = builder.addInputTensor(popart.TensorInfo("INT32", label_shape))

    proto = builder.getModelProto()

    losses = [popart.NllLoss(output, label, "NllLossVal")]

    # Loss on the last IPU
    losses[0].virtualGraph(1 if enable_multi_ipu else 0)

    art = popart.AnchorReturnType("ALL")
    anchorNames = {losses[0].output(0): art}

    if return_stepio:
        anchorNames[popart.reservedGradientPrefix() + w0] = art
        anchorNames[popart.reservedAccumulationPrefix() +
                    popart.reservedGradientPrefix() + w0] = art
        anchorNames[popart.reservedAccumulationOutPrefix() +
                    popart.reservedGradientPrefix() + w0] = art

    opts = popart.SessionOptions()
    opts.enableGradientAccumulation = enable_accl
    opts.accumulationFactor = accl_steps
    opts.enableOutlining = False
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    if enable_multi_ipu:
        device = tu.get_ipu_model(numIPUs=num_ipus, compileIPUCode=False)
    else:
        device = tu.get_ipu_model(compileIPUCode=False)

    session = popart.TrainingSession(fnModel=proto,
                                     dataFeed=popart.DataFlow(
                                         batches_per_step, anchorNames),
                                     deviceInfo=device,
                                     losses=losses,
                                     optimizer=popart.SGD(learning_rate=0.1,
                                                          weight_decay=0.1),
                                     userOptions=opts)

    session.prepareDevice()
    session.weightsFromHost()
    session.optimizerFromHost()

    anchorArrays = session.initAnchorArrays()

    outer_dim = 1
    if batches_per_step > 1:
        outer_dim *= batches_per_step
        labelArray = np.repeat(labelArray[np.newaxis], batches_per_step, 0)
    if accl_steps > 1:
        outer_dim *= accl_steps
        labelArray = labelArray.reshape([accl_steps * batches_per_step, -1])
    if outer_dim > 1:
        input_shape = [outer_dim] + input_shape

    stepio = popart.PyStepIO(
        {
            input_: np.ones(input_shape, np.float32),
            label: labelArray.astype(np.int32)
        }, anchorArrays)

    if return_stepio:
        return session, stepio, anchorArrays, w0

    for i in range(6):
        session.run(stepio)

    proto_file = "{}.onnx".format(proto_name)
    session.modelToHost(proto_file)
    return proto_file


def check_models(modelA, modelB, epsilon=1e-9):
    modelA = onnx.load(modelA)
    modelB = onnx.load(modelB)

    for w_i, weightA in enumerate(modelA.graph.initializer):
        for d_i, dataA in enumerate(weightA.float_data):
            dataB = modelB.graph.initializer[w_i].float_data[d_i]
            assert epsilon > abs(dataA - dataB), "Difference {}".format(dataA -
                                                                        dataB)


def test_gradient_accumulation_1():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, batch_size)
    accl_proto = run_graph(labelArray, 4, True, 1, "acclNoSteps", False)
    norm_proto = run_graph(labelArray, 1, False, 1, "NoacclNoSteps", False)

    check_models(accl_proto, norm_proto)


def test_gradient_accumulation_2():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, batch_size)
    accl_proto = run_graph(labelArray, 4, True, 5, "acclSteps", False)
    norm_proto = run_graph(labelArray, 1, False, 5, "NoacclSteps", False)

    check_models(accl_proto, norm_proto)


def test_gradient_accumulation_multi_ipu():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, batch_size)
    accl_proto = run_graph(labelArray, 4, True, 5, "acclSteps", True)
    norm_proto = run_graph(labelArray, 1, False, 5, "NoacclSteps", True)

    check_models(accl_proto, norm_proto)


def test_gradient_accumulation_error_1():
    builder = popart.Builder()

    input_shape = [1, 64]
    input_ = builder.addInputTensor(popart.TensorInfo("FLOAT16", input_shape))

    x = input_
    for i in range(2):
        w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
        x = builder.aiOnnx.matmul([x, w])
    output = x
    builder.addOutputTensor(output)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {output: popart.AnchorReturnType("FINAL")})

    opts = popart.SessionOptions()
    opts.enableGradientAccumulation = True
    opts.accumulationFactor = 10

    device = tu.get_ipu_model(compileIPUCode=False)

    with pytest.raises(popart.popart_exception) as e_info:
        popart.InferenceSession(fnModel=proto,
                                dataFeed=dataFlow,
                                userOptions=opts,
                                deviceInfo=device)

    assert e_info.value.args[0].startswith(
        "Gradient Accumulation only available when training")


def test_gradient_accumulation_error_2():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, batch_size)

    with pytest.raises(popart.popart_exception) as e_info:
        # This combo of options will throw an error (accl factor > 1 but accl disabled.)
        run_graph(labelArray, 4, False, 1, "acclNoSteps", False)

    assert e_info.value.args[0].startswith(
        "enableGradientAccumulation is false, but accumulationFactor > 1.")


def test_gradient_accumulation_anchors():
    """Return the gradients of a linear operation and see if a multiple of
    the gradient is equal to the accumulated gradient.
    """

    np.random.seed(1234)
    accl_steps = 4
    labelArray = np.random.randint(0, hidden_size, batch_size)
    session, stepio, anchors, w0 = run_graph(labelArray,
                                             accl_steps,
                                             True,
                                             1,
                                             "acclNoSteps",
                                             False,
                                             return_stepio=True)
    session.run(stepio)
    # Beacause the matmuls are linear, we can just do grad_accl steps * weight_delta
    # to get the accumulated gradient tensor.
    assert np.allclose(
        (anchors[popart.reservedGradientPrefix() + w0]) * accl_steps,
        (anchors[popart.reservedAccumulationOutPrefix() +
                 popart.reservedGradientPrefix() + w0]))


def run_graph_complex(labelArray,
                      accl_steps,
                      enable_accl,
                      batches_per_step,
                      proto_name,
                      enable_multi_ipu,
                      return_stepio=False):
    micro_batch_size = batch_size // accl_steps

    builder = popart.Builder()

    input_shape = [micro_batch_size, 2, 4, 4]
    input_ = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape))
    w0 = builder.addInitializedInputTensor(
        np.ones([2, 2, 3, 3]).astype(np.float32))
    x0 = input_

    s0 = builder.aiOnnx.sin([x0], "s0")
    e0 = builder.aiOnnx.exp([s0], "e0")

    c0 = builder.aiOnnx.conv([e0, w0],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1],
                             debugPrefix="c0")
    r0 = builder.reshape_const(builder.aiOnnx, [c0], [micro_batch_size, 32])
    output = builder.aiOnnx.softmax([r0], axis=1, debugPrefix="sfm")

    builder.addOutputTensor(output)

    label_shape = [micro_batch_size]
    label = builder.addInputTensor(popart.TensorInfo("INT32", label_shape))

    proto = builder.getModelProto()

    losses = [popart.NllLoss(output, label, "NllLossVal")]

    art = popart.AnchorReturnType("ALL")
    anchorNames = {losses[0].output(0): art}

    if return_stepio:
        anchorNames[popart.reservedGradientPrefix() + w0] = art
        anchorNames[popart.reservedAccumulationPrefix() +
                    popart.reservedGradientPrefix() + w0] = art
        anchorNames[popart.reservedAccumulationOutPrefix() +
                    popart.reservedGradientPrefix() + w0] = art

    opts = popart.SessionOptions()
    opts.enableGradientAccumulation = enable_accl
    opts.accumulationFactor = accl_steps
    opts.enableOutlining = False
    opts.virtualGraphMode = popart.VirtualGraphMode.Off

    if enable_multi_ipu:
        device = tu.get_ipu_model(numIPUs=num_ipus, compileIPUCode=False)
    else:
        device = tu.get_ipu_model(compileIPUCode=False)

    session = popart.TrainingSession(fnModel=proto,
                                     dataFeed=popart.DataFlow(
                                         batches_per_step, anchorNames),
                                     deviceInfo=device,
                                     losses=losses,
                                     optimizer=popart.SGD(learning_rate=0.1,
                                                          weight_decay=0.1),
                                     userOptions=opts)

    session.prepareDevice()
    session.weightsFromHost()
    session.optimizerFromHost()

    anchorArrays = session.initAnchorArrays()

    outer_dim = 1
    if batches_per_step > 1:
        outer_dim *= batches_per_step
        labelArray = np.repeat(labelArray[np.newaxis], batches_per_step, 0)
    if accl_steps > 1:
        outer_dim *= accl_steps
        labelArray = labelArray.reshape([accl_steps * batches_per_step, -1])
    if outer_dim > 1:
        input_shape = [outer_dim] + input_shape

    stepio = popart.PyStepIO(
        {
            input_: np.ones(input_shape, np.float32),
            label: labelArray.astype(np.int32)
        }, anchorArrays)

    if return_stepio:
        return session, stepio, anchorArrays, w0

    session.run(stepio)

    proto_file = "{}.onnx".format(proto_name)
    session.modelToHost(proto_file)
    return proto_file


def test_gradient_accumulation_complex_1():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, batch_size)
    accl_proto = run_graph_complex(labelArray, 4, True, 1, "acclNoSteps",
                                   False)
    norm_proto = run_graph_complex(labelArray, 1, False, 1, "NoacclNoSteps",
                                   False)

    # Gradient accumulation will cause weights to diverge slightly, especially over
    # more than one step so we only test if they are reasonably close.
    check_models(accl_proto, norm_proto, epsilon=1e-2)


def test_gradient_accumulation_complex_2():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, batch_size)
    accl_proto = run_graph_complex(labelArray, 4, True, 5, "acclSteps", False)
    norm_proto = run_graph_complex(labelArray, 1, False, 5, "NoacclSteps",
                                   False)

    check_models(accl_proto, norm_proto, epsilon=1e-2)
