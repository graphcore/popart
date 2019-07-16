import numpy as np
import onnx
import pytest
import poponnx
import test_util as tu

true_batch_size = 16
hidden_size = 16


def run_graph(labelArray, accl_steps, enable_accl, batches_per_step,
              proto_name):
    batch_size = true_batch_size // accl_steps

    builder = poponnx.Builder()

    input_shape = [batch_size, hidden_size]
    input_ = builder.addInputTensor(poponnx.TensorInfo("FLOAT", input_shape))

    x = input_
    for i in range(6):
        w = builder.addInitializedInputTensor(
            np.ones([hidden_size, hidden_size]).astype(np.float32))
        x = builder.aiOnnx.matmul([x, w])
    output = x

    builder.addOutputTensor(output)

    label_shape = [batch_size]
    label = builder.addInputTensor(poponnx.TensorInfo("INT32", label_shape))

    proto = builder.getModelProto()

    losses = [poponnx.NllLoss(output, label, "NllLossVal")]

    anchorNames = {
        losses[0].output(0): poponnx.AnchorReturnType("ALL"),
    }

    opts = poponnx.SessionOptions()
    opts.enableGradientAccumulation = enable_accl
    opts.accumulationFactor = accl_steps
    opts.enableOutlining = False

    device = tu.get_ipu_model(compileIPUCode=False)

    session = poponnx.TrainingSession(fnModel=proto,
                                      dataFeed=poponnx.DataFlow(
                                          batches_per_step, anchorNames),
                                      deviceInfo=device,
                                      losses=losses,
                                      optimizer=poponnx.SGD(learning_rate=0.1,
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

    for i in range(6):
        stepio = poponnx.PyStepIO(
            {
                input_: np.ones(input_shape, np.float32),
                label: labelArray.astype(np.int32)
            }, anchorArrays)
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
            assert dataA == dataB and epsilon > abs(
                dataA - dataB), "Difference {}".format(dataA - dataB)


def test_gradient_accumulation_1():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, true_batch_size)
    accl_proto = run_graph(labelArray, 4, True, 1, "acclNoSteps")
    norm_proto = run_graph(labelArray, 1, False, 1, "NoacclNoSteps")

    check_models(accl_proto, norm_proto)


def test_gradient_accumulation_2():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, true_batch_size)
    accl_proto = run_graph(labelArray, 4, True, 5, "acclSteps")
    norm_proto = run_graph(labelArray, 1, False, 5, "NoacclSteps")

    check_models(accl_proto, norm_proto)


def test_gradient_accumulation_error_1():
    builder = poponnx.Builder()

    input_shape = [1, 64]
    input_ = builder.addInputTensor(poponnx.TensorInfo("FLOAT16", input_shape))

    x = input_
    for i in range(2):
        w = builder.addInitializedInputTensor(np.zeros([64, 64], np.float16))
        x = builder.aiOnnx.matmul([x, w])
    output = x
    builder.addOutputTensor(output)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {output: poponnx.AnchorReturnType("FINAL")})

    opts = poponnx.SessionOptions()
    opts.enableGradientAccumulation = True
    opts.accumulationFactor = 10

    device = tu.get_ipu_model(compileIPUCode=False)

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.InferenceSession(fnModel=proto,
                                 dataFeed=dataFlow,
                                 userOptions=opts,
                                 deviceInfo=device)

    assert (e_info.value.args[0].startswith(
        "Gradient Accumulation only available when training"))


def test_gradient_accumulation_error_2():
    np.random.seed(1234)
    labelArray = np.random.randint(0, hidden_size, true_batch_size)

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        # This combo of options will throw an error (accl factor > 1 but accl disabled.)
        run_graph(labelArray, 4, False, 1, "acclNoSteps")

    assert (e_info.value.args[0].startswith(
        "enableGradientAccumulation is false, but accumulationFactor > 1."))
