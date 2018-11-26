import numpy as np
import pytest
import poponnx
import test_util as tu


def test_convolution_cached_by_default():
    """
    In this test we check that the default behaviour is for convolutions to be
    cached.
    """

    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = poponnx.TensorInfo("FLOAT", [2, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    c1 = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    o = builder.convolution([c1, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)
    earlyInfo.add(i2, filt_shape)
    anchor_names = ['d__' + i1, 'd__' + i2]
    dataFlow = poponnx.DataFlow(1, 1, anchor_names,
                                poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.reportOptions = {"doLayerWiseBreakdown": "true"}

    passes = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts,
        passes=passes,
        outputdir="/tmp")

    session.setDevice(tu.get_ipu_model(compileIPUCode=False))
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.train(stepio)
    session.weightsFromHost()
    session.optimizerFromHost()

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_3x3_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_3x3/Convolve$', computeSets)
    num_4x4_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_4x4/Convolve$', computeSets)
    # There should be only one convolution of each type
    assert (num_3x3_convolutions == 1)
    assert (num_4x4_convolutions == 1)


def test_convolution_cached_set_to_true():
    """
    In this test we check that the convolutions are cached when they asked to be
    cached.
    """

    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = poponnx.TensorInfo("FLOAT", [2, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    c1 = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1],
                             1,
                             cacheOperation=True)
    o = builder.convolution([c1, i2], [1, 1], [1, 1, 1, 1], [1, 1],
                            1,
                            cacheOperation=True)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)
    earlyInfo.add(i2, filt_shape)
    anchor_names = ['d__' + i1, 'd__' + i2]
    dataFlow = poponnx.DataFlow(1, 1, anchor_names,
                                poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.reportOptions = {"doLayerWiseBreakdown": "true"}

    passes = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts,
        passes=passes,
        outputdir="/tmp")

    session.setDevice(tu.get_ipu_model(compileIPUCode=False))
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.train(stepio)
    session.weightsFromHost()
    session.optimizerFromHost()

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_3x3_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_3x3/Convolve$', computeSets)
    num_4x4_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_4x4/Convolve$', computeSets)
    # There should be only one convolution of each type
    assert (num_3x3_convolutions == 1)
    assert (num_4x4_convolutions == 1)


def test_convolution_cached_set_to_false():
    """
    In this test we check that the convolutions are not cached when they asked
    to not be cached.
    """

    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = poponnx.TensorInfo("FLOAT", [2, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    c1 = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1],
                             1,
                             cacheOperation=False)
    o = builder.convolution([c1, i2], [1, 1], [1, 1, 1, 1], [1, 1],
                            1,
                            cacheOperation=False)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)
    earlyInfo.add(i2, filt_shape)
    anchor_names = ['d__' + i1, 'd__' + i2]
    dataFlow = poponnx.DataFlow(1, 1, anchor_names,
                                poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.reportOptions = {"doLayerWiseBreakdown": "true"}

    passes = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts,
        passes=passes,
        outputdir="/tmp")

    session.setDevice(tu.get_ipu_model(compileIPUCode=False))
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.train(stepio)
    session.weightsFromHost()
    session.optimizerFromHost()

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_3x3_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_3x3/Convolve$', computeSets)
    num_4x4_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_4x4/Convolve$', computeSets)
    # Two 3x3 convolutions (bwd + fwd) for each convolution in the graph
    assert (num_3x3_convolutions == 4)
    # Two updates
    assert (num_4x4_convolutions == 2)


def test_convolution_some_convolutions_cached():
    """
    In this test we check that the correctness of having some convolutions
    cached and some not.
    """

    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = poponnx.TensorInfo("FLOAT", [2, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    c1 = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    c2 = builder.convolution([c1, i2], [1, 1], [1, 1, 1, 1], [1, 1],
                             1,
                             cacheOperation=False)
    o = builder.convolution([c2, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)
    earlyInfo.add(i2, filt_shape)
    anchor_names = ['d__' + i1, 'd__' + i2]
    dataFlow = poponnx.DataFlow(1, 1, anchor_names,
                                poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.reportOptions = {"doLayerWiseBreakdown": "true"}

    passes = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts,
        passes=passes,
        outputdir="/tmp")

    session.setDevice(tu.get_ipu_model(compileIPUCode=False))
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.train(stepio)
    session.weightsFromHost()
    session.optimizerFromHost()

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_3x3_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_3x3/Convolve$', computeSets)
    num_4x4_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_4x4/Convolve$', computeSets)
    # Two 3x3 convolutions (bwd + fwd) for the uncached convolution in the graph and then 1 for all the others
    assert (num_3x3_convolutions == 3)
    # One cached and one uncached update
    assert (num_4x4_convolutions == 2)


def test_convolution_disable_all():
    """
    In this test we check that the correctness of having some convolutions
    cached and some not.
    """

    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = poponnx.TensorInfo("FLOAT", [2, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    c1 = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1],
                             1,
                             cacheOperation=True)
    c2 = builder.convolution([c1, i2], [1, 1], [1, 1, 1, 1], [1, 1],
                             1,
                             cacheOperation=True)
    o = builder.convolution([c2, i2], [1, 1], [1, 1, 1, 1], [1, 1],
                            1,
                            cacheOperation=True)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    earlyInfo = poponnx.EarlyInfo()
    earlyInfo.add(i1, data_shape)
    earlyInfo.add(i2, filt_shape)
    anchor_names = ['d__' + i1, 'd__' + i2]
    dataFlow = poponnx.DataFlow(1, 1, anchor_names,
                                poponnx.AnchorReturnType.ALL)
    optimizer = poponnx.SGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]

    opts = poponnx.SessionOptionsCore()
    opts.reportOptions = {"doLayerWiseBreakdown": "true"}
    opts.enableConvolutionGraphCaching = False

    passes = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]

    session = poponnx.Session(
        fnModel=proto,
        earlyInfo=earlyInfo,
        dataFeed=dataFlow,
        losses=losses,
        optimizer=optimizer,
        userOptions=opts,
        passes=passes,
        outputdir="/tmp")

    session.setDevice(tu.get_ipu_model(compileIPUCode=False))
    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = poponnx.PyStepIO(inputs, anchors)

    session.train(stepio)
    session.weightsFromHost()
    session.optimizerFromHost()

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_3x3_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_3x3/Convolve$', computeSets)
    num_4x4_convolutions = tu.get_compute_set_regex_count(
        r'^[0-9]+/Conv_4x4/Convolve$', computeSets)
    # Two 3x3 convolutions (bwd + fwd) for each convolution
    assert (num_3x3_convolutions == 6)
    # Updates
    assert (num_4x4_convolutions == 3)
