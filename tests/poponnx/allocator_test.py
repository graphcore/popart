import numpy as np
import pytest
import poponnx
import test_util as tu


# model:
#
#  i1 -- Squeeze -- |
#                   |-- Conv --> o
#  i2 -- Squeeze -- |
#
def test_allocator_conv_control():
    """
    In this test we check that a convolution does not allocate an input
    tensor when 'unwinding' through 'deadend' ops (in this case Squeeze)

    We observe that the poplar compute sets for this graph include a 
    'PreArrange' as a result
    """

    builder = poponnx.Builder()

    data_input_shape = poponnx.TensorInfo("FLOAT", [1, 1, 2, 4, 4])
    filt_input_shape = poponnx.TensorInfo("FLOAT", [1, 2, 2, 3, 3])

    i1 = builder.addInputTensor(data_input_shape)
    s1 = builder.squeeze([i1], [0])  # shape = [1, 2, 4, 4]

    i2 = builder.addInputTensor(filt_input_shape)
    s2 = builder.squeeze([i2], [0])  # shape = [2, 2, 3, 3]

    o = builder.convolution([s1, s2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})
    optimizer = poponnx.ConstSGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.reportOptions = {"doLayerWiseBreakdown": "true"}

    session = poponnx.Session(
        fnModel=proto,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts)

    session.setDevice(tu.get_ipu_model(compileIPUCode=False))
    anchors = session.initAnchorArrays()

    session.prepareDevice()
    session.weightsFromHost()

    data = np.ones(data_input_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_input_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.train(stepio)

    # Check for the pre-arrange step before the convolution
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)
    num_prearrange = tu.get_compute_set_regex_count(r'Convolve/PreArrange',
                                                    computeSets)
    assert (num_prearrange == 1)


# model:
#
#  i1 --|-- Reshape --|-- Transpose --|
#                                     |-- Conv --> o
#  i2 --|-- Reshape --|-- Transpose --|
#
def test_allocator_conv():
    """
    In this test we check that a convolution can allocate an input
    tensor by 'unwinding' through ops (transpose and reshape) that
    have non-identity unwind functions

    We observe that the poplar compute sets for this graph do not
    include a 'PreArrange', unlike in the control test 
    """

    builder = poponnx.Builder()

    # data
    data_input_shape = poponnx.TensorInfo("FLOAT", [1, 32])
    i1 = builder.addInputTensor(data_input_shape)
    r1 = builder.reshape_const([i1], [4, 4, 2, 1], "r1")
    t1 = builder.transpose([r1], [3, 2, 1, 0], "t1")  # shape = [1, 2, 4, 4]

    # weights
    filt_input_shape = poponnx.TensorInfo("FLOAT", [36])
    i2 = builder.addInputTensor(filt_input_shape)
    r2 = builder.reshape_const([i2], [3, 3, 2, 2], "r2")
    t2 = builder.transpose([r2], [3, 2, 1, 0], "t2")  # shape = [2, 2, 3, 3]

    # conv
    o = builder.convolution([t1, t2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})
    # dataFlow = poponnx.DataFlow(1, {})
    optimizer = poponnx.ConstSGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.reportOptions = {"doLayerWiseBreakdown": "true"}

    session = poponnx.Session(
        fnModel=proto,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts)

    session.setDevice(tu.get_ipu_model(compileIPUCode=False))
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_input_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_input_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.train(stepio)

    # Check there are no pre-arranges before the convolution
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)
    num_prearrange = tu.get_compute_set_regex_count(r'Convolve/PreArrange',
                                                    computeSets)
    assert (num_prearrange == 0)
