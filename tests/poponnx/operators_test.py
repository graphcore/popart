import numpy as np
import pytest
import poponnx
import torch
import test_util as tu


def test_get_op_types():
    ops_public = poponnx.getSupportedOperations(False)
    assert (len(ops_public) > 0)

    ops_all = poponnx.getSupportedOperations(True)
    assert (len(ops_all) > 0)
    assert (len(ops_all) > len(ops_public))


def test_add(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, shape)
    earlyInfo.add(i2, shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1: np.array([1, 2], dtype=np.float32),
        i2: np.array([3, 4], dtype=np.float32)
    }
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    assert (np.array_equal(anchors[o], [4, 6]))


def test_convolution(tmpdir):

    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = poponnx.TensorInfo("FLOAT", [3, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    o = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)
    earlyInfo.add(i2, filt_shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    expected = np.array([[[[8., 12., 12., 8.], [12., 18., 18., 12.],
                           [12., 18., 18., 12.], [8., 12., 12., 8.]],
                          [[8., 12., 12., 8.], [12., 18., 18., 12.],
                           [12., 18., 18., 12.], [8., 12., 12., 8.]],
                          [[8., 12., 12., 8.], [12., 18., 18., 12.],
                           [12., 18., 18., 12.], [8., 12., 12., 8.]]]],
                        dtype=np.float32)

    assert (np.array_equal(anchors[o], expected))


def test_matmul(tmpdir):
    # create a basic model with a matmul operator
    # and compare the answer against numpy.matmul

    builder = poponnx.Builder()

    i1shape = poponnx.TensorInfo("FLOAT", [2, 3])
    i2shape = poponnx.TensorInfo("FLOAT", [3, 4])

    i1 = builder.addInputTensor(i1shape)
    i2 = builder.addInputTensor(i2shape)
    o = builder.matmul([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, i1shape)
    earlyInfo.add(i2, i2shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    inputs = {
        i1:
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        i2:
        np.array([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]],
                 dtype=np.float32)
    }

    stepio = poponnx.PyStepIO(inputs, anchors)

    session.infer(stepio)

    # test the poponnx answer against the numpy answer
    assert (np.array_equal(anchors[o], np.matmul(inputs[i1], inputs[i2])))


def test_averagepool(tmpdir):
    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 1, 6, 6])
    i1 = builder.addInputTensor(data_shape)

    o = builder.averagepool([i1], [2, 2], [2, 2], [0, 0, 0, 0])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.random.rand(1, 1, 6, 6).astype(np.float32)

    inputs = {i1: data}

    stepio = poponnx.PyStepIO(inputs, anchors)
    session.infer(stepio)

    # get the pytorch output
    torch_avgpool = torch.nn.AvgPool2d(2, 2)
    torch_output = torch_avgpool(torch.from_numpy(data))

    assert (np.array_equal(torch_output.numpy(), anchors[o]))


def test_maxpool(tmpdir):
    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 1, 6, 6])
    i1 = builder.addInputTensor(data_shape)

    o = builder.maxpool([i1], [2, 2], [2, 2], [0, 0, 0, 0])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        outputdir=str(tmpdir))

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.random.rand(1, 1, 6, 6).astype(np.float32)

    inputs = {i1: data}

    stepio = poponnx.PyStepIO(inputs, anchors)
    session.infer(stepio)

    # get the pytorch output
    torch_avgpool = torch.nn.MaxPool2d(2, 2)
    torch_output = torch_avgpool(torch.from_numpy(data))

    assert (np.array_equal(torch_output.numpy(), anchors[o]))


def test_mul(tmpdir):
    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [2])
    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)

    o = builder.mul([i1, i2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, shape)
    earlyInfo.add(i2, shape)

    print('')
    print('i1', i1)
    print('i2', i2)
    print('o', o)
    dataFlow = poponnx.DataFlow(1, 1, [o, 'd__' + i1, 'd__' + i2, 'd__' + o],
                                poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.logging = {"all": "TRACE"}
    opts.exportDot = True

    optPasses = poponnx.Patterns(["PreUniRepl", "MulArgGradOp"])

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir),
        passes=optPasses,
        userOptions=opts)

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    # generate random test data
    d1 = np.random.rand(*shape.shape()).astype(np.float32)
    d2 = np.random.rand(*shape.shape()).astype(np.float32)

    inputs = {i1: d1, i2: d2}

    # run poponnx training step
    stepio = poponnx.PyStepIO(inputs, anchors)
    session.train(stepio)

    # do a multiply with pytorch to compare to poponnx output
    ti1 = torch.from_numpy(d1)
    ti1.requires_grad_(True)
    ti2 = torch.from_numpy(d2)
    ti2.requires_grad_(True)
    to = ti1 * ti2

    # pass poponnx output gradient through the pytorch mul gradient
    # to compare to poponnx mul gradient output
    tg = to.grad_fn(torch.from_numpy(anchors['d__' + o]))

    # convert torch output to numpy
    to = to.data.numpy()
    tg = (tg[0].data.numpy(), tg[1].data.numpy())

    # check output of op
    assert np.array_equal(to, anchors[o])
    # check output of gradient ops
    assert np.array_equal(tg[0], anchors['d__' + i1])
    assert np.array_equal(tg[1], anchors['d__' + i2])


def test_broadcast_mul(tmpdir):
    builder = poponnx.Builder()

    i1_shape = poponnx.TensorInfo("FLOAT", [2, 2])
    i2_shape = poponnx.TensorInfo("FLOAT", [2])

    i1 = builder.addInputTensor(i1_shape)
    i2 = builder.addInputTensor(i2_shape)

    o = builder.mul([i1, i2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, i1_shape)
    earlyInfo.add(i2, i2_shape)

    print('')
    print('i1', i1)
    print('i2', i2)
    print('o', o)
    dataFlow = poponnx.DataFlow(1, 1, [o, 'd__' + i1, 'd__' + i2, 'd__' + o],
                                poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.logging = {"all": "TRACE"}
    opts.exportDot = True

    optPatterns = poponnx.Patterns(["PreUniRepl", "MulArgGradOp"])

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        outputdir=str(tmpdir),
        passes=optPatterns,
        userOptions=opts)

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    # generate random test data
    d1 = np.random.rand(*i1_shape.shape()).astype(np.float32)
    d2 = np.random.rand(*i2_shape.shape()).astype(np.float32)

    inputs = {i1: d1, i2: d2}

    # run poponnx training step
    stepio = poponnx.PyStepIO(inputs, anchors)
    session.train(stepio)

    # do a multiply with pytorch to compare to poponnx output
    ti1 = torch.from_numpy(d1)
    ti1.requires_grad_(True)
    ti2 = torch.from_numpy(d2)
    ti2.requires_grad_(True)
    to = ti1 * ti2

    # pass poponnx output gradient through the pytorch mul gradient
    # to compare to poponnx mul gradient output
    tg = to.grad_fn(torch.from_numpy(anchors['d__' + o]))

    # convert torch output to numpy
    to = to.data.numpy()
    tg = (tg[0].data.numpy(), tg[1].data.numpy())

    # check output of op
    assert np.array_equal(to, anchors[o])
    # check the grad tensors shapes match the input tensors
    assert (d1.shape == anchors['d__' + i1].shape)
    assert (d2.shape == anchors['d__' + i2].shape)
    # check output of gradient ops
    # need to do a reduce sum on one of the pytorch outputs
    assert np.array_equal(tg[0], anchors['d__' + i1])
    assert np.array_equal(np.sum(tg[1], axis=0), anchors['d__' + i2])


def test_reciprocal(tmpdir):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    # create graph
    test = tu.BasicSession(tmpdir)
    i1 = test.add_input_tensor(d1)
    o = test.builder.reciprocal([i1])
    test.builder.addOutputTensor(o)

    test.passes.extend(["PreUniRepl"])

    # run the poponnx session
    anchors = test.run(o, [o], 'infer')

    # create and run numpy reference
    def numpy_reference(i1):
        outputs = {}
        outputs[o] = 1 / i1
        return outputs

    reference_results = numpy_reference(d1)

    # compare results
    for key in [o]:
        print('Checking anchor %s ...' % (key, ))
        assert np.array_equal(anchors[key], reference_results[key])


def test_reciprocal_grad(tmpdir):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    # create graph
    test = tu.BasicSession(tmpdir)
    i1 = test.add_input_tensor(d1)
    o = test.builder.reciprocal([i1])
    test.builder.addOutputTensor(o)

    test.passes.extend(["PreUniRepl", "ReciprocalGradOp"])

    # run the poponnx session
    anchors = test.run(o, [o, 'd__' + o, 'd__' + i1], 'train')

    # create and run numpy reference
    def torch_reference(i, d__o):
        a = torch.tensor(i)
        a.requires_grad_(True)
        b = 1 / a
        b.backward(torch.tensor(d__o))

        outputs = {}
        outputs[o] = b.data.numpy()
        outputs['d__' + i1] = a.grad.data.numpy()

        return outputs

    reference_results = torch_reference(d1, anchors['d__' + o])

    # compare results
    for key in [o, 'd__' + i1]:
        print('Checking anchor %s ...' % (key, ))
        assert np.allclose(anchors[key], reference_results[key])
