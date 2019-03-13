import poponnx
import pytest


# Both manual and auto recomputation are not supported
# concurrently
def test_valid_recompute_options():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1]))
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    # specify manual recomputation
    builder.recomputeOutputInBackwardPass(o)

    # specify auto recomputation as well
    opts = poponnx.SessionOptions()
    opts.enableAutoRecomputation = True

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session = poponnx.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")}),
            optimizer=poponnx.ConstSGD(0.001),
            losses=[poponnx.L1Loss(o, "l1LossVal", 0.1)],
            passes=poponnx.Patterns([]),
            userOptions=opts)
    assert (
        e_info.value.args[0] ==
        "A mixture of auto and manual recomputaion is currently not supported")
