import numpy as np
import poponnx
import test_util as tu


def test_constants_preserved():
    # Check that `session.modelToHost` can be called when using a
    # model with a constant node, without throwing an exceptions
    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [2, 2]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [2, 2]))
    c = builder.constant(np.array([[1., 6.], [4., 5.]], dtype=np.float32))
    o1 = builder.add([i1, i2])
    o2 = builder.add([o1, c])
    o = o2
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    anchors = {o: poponnx.AnchorReturnType("ALL")}

    dataFlow = poponnx.DataFlow(1, anchors)

    optimizer = poponnx.ConstSGD(0.01)

    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}

    session = poponnx.Session(
        fnModel=proto,
        dataFeed=dataFlow,
        userOptions=opts,
        losses=losses,
        optimizer=optimizer)
    session.setDevice(tu.get_poplar_cpu_device())

    anchorArrays = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1: np.random.rand(2, 2).astype(np.float32),
        i2: np.random.rand(2, 2).astype(np.float32),
    }
    pystepio = poponnx.PyStepIO(inputs, anchorArrays)
    session.train(pystepio)

    session.modelToHost('session_proto.onnx')

    # models should be the same after training
    # as there are no trainable parameters
    with open('session_proto.onnx', 'rb') as f:
        session_proto = f.read()
    assert proto == session_proto
