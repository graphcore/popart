# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu


@tu.requires_ipu_model
def test_convolution_cached_by_default():
    """
    In this test we check that the default behavior is for convolutions to be
    cached.
    """

    builder = popart.Builder()

    data_shape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = popart.TensorInfo("FLOAT", [2, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    c1 = builder.aiOnnx.conv([i1, i2],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1])
    o = builder.aiOnnx.conv([c1, i2],
                            dilations=[1, 1],
                            pads=[1, 1, 1, 1],
                            strides=[1, 1])
    loss = builder.aiGraphcore.identityloss([o])

    proto = builder.getModelProto()

    anchor_names = [
        popart.reservedGradientPrefix() + i1,
        popart.reservedGradientPrefix() + i2
    ]
    dataFlow = \
        popart.DataFlow(1,
                         {anchor_names[0] : popart.AnchorReturnType("All"),
                          anchor_names[1] : popart.AnchorReturnType("All")})
    optimizer = popart.ConstSGD(0.01)

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=dataFlow,
        loss=loss,
        optimizer=optimizer,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)
    session.weightsFromHost()

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)
    num_3x3_convolutions = tu.get_compute_set_regex_count(
        r'^.+/convolution/Conv_3x3/Convolve$', computeSets)
    num_4x4_convolutions = tu.get_compute_set_regex_count(
        r'^.+/weightDeltas/Conv_4x4/Convolve$', computeSets)
    # There should be only one convolution of each type
    assert (num_3x3_convolutions == 1)
    assert (num_4x4_convolutions == 1)


@tu.requires_ipu_model
def test_convolution_disable_all():
    """
    In this test we check that the correctness of having some convolutions
    cached and some not.
    """

    builder = popart.Builder()

    data_shape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = popart.TensorInfo("FLOAT", [2, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    c1 = builder.aiOnnx.conv([i1, i2],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1])
    c2 = builder.aiOnnx.conv([c1, i2],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1])
    o = builder.aiOnnx.conv([c2, i2],
                            dilations=[1, 1],
                            pads=[1, 1, 1, 1],
                            strides=[1, 1])
    loss = builder.aiGraphcore.identityloss([o])

    proto = builder.getModelProto()

    anchor_names = [
        popart.reservedGradientPrefix() + i1,
        popart.reservedGradientPrefix() + i2
    ]
    dataFlow = \
        popart.DataFlow(1,
                         {anchor_names[0] : popart.AnchorReturnType("All"),
                          anchor_names[1] : popart.AnchorReturnType("All")})
    optimizer = popart.ConstSGD(0.01)

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enableOutlining = False

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=dataFlow,
        loss=loss,
        optimizer=optimizer,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    data = np.ones(data_shape.shape(), dtype=np.float32)
    filt = np.ones(filt_shape.shape(), dtype=np.float32)

    inputs = {i1: data, i2: filt}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)
    session.weightsFromHost()

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_3x3_convolutions = tu.get_compute_set_regex_count(
        r'^.+/convolution/Conv_3x3/Convolve$', computeSets)
    num_4x4_convolutions = tu.get_compute_set_regex_count(
        r'^.+/weightDeltas/Conv_4x4/Convolve$', computeSets)
    # Two 3x3 convolutions (bwd + fwd) for each convolution
    assert (num_3x3_convolutions == 6)
    # Updates
    assert (num_4x4_convolutions == 3)


@tu.requires_ipu_model
def test_matmul_infer_cached_by_default():
    """
    In this test we check that the default behaviour is for matmul to be
    cached.
    """

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    matmul_lhs_shape = popart.TensorInfo("FLOAT", [2, 3])
    matmul_rhs_shape = popart.TensorInfo("FLOAT", [3, 4])

    i1 = builder.addInputTensor(matmul_lhs_shape)
    i2 = builder.addInputTensor(matmul_rhs_shape)
    i3 = builder.addInputTensor(matmul_lhs_shape)
    i4 = builder.addInputTensor(matmul_rhs_shape)

    c1 = builder.aiOnnx.matmul([i1, i2])
    c2 = builder.aiOnnx.matmul([i3, i4])

    a1 = builder.aiOnnx.add([c1, c2])

    c3 = builder.aiOnnx.matmul([i1, i2])
    c4 = builder.aiOnnx.matmul([i3, i4])

    a2 = builder.aiOnnx.add([c3, c4])

    o = builder.aiOnnx.add([a1, a2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    anchor_names = [o]
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    matmul1_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul1_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    matmul2_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul2_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    inputs = {
        i1: matmul1_lhs,
        i2: matmul1_rhs,
        i3: matmul2_lhs,
        i4: matmul2_rhs
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_matmuls = tu.get_compute_set_regex_count(
        r'^.+/matmulGrouped/Conv_1/Convolve$', computeSets)
    # There should be only one matmul
    assert (num_matmuls == 1)


@tu.requires_ipu_model
def test_matmul_train_cached_by_default():
    """
    In this test we check that the default behaviour is for matmul to be
    cached.
    """

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    matmul1_lhs_shape = popart.TensorInfo("FLOAT", [2, 3])
    matmul1_rhs_shape = popart.TensorInfo("FLOAT", [3, 4])

    matmul2_lhs_shape = popart.TensorInfo("FLOAT", [2, 3])
    matmul2_rhs_shape = popart.TensorInfo("FLOAT", [3, 4])

    i1 = builder.addInputTensor(matmul1_lhs_shape)
    i2 = builder.addInputTensor(matmul1_rhs_shape)
    i3 = builder.addInputTensor(matmul2_lhs_shape)
    i4 = builder.addInputTensor(matmul2_rhs_shape)

    c1 = builder.aiOnnx.matmul([i1, i2])
    c2 = builder.aiOnnx.matmul([i3, i4])

    a1 = builder.aiOnnx.add([c1, c2])

    c3 = builder.aiOnnx.matmul([i1, i2])
    c4 = builder.aiOnnx.matmul([i3, i4])

    a2 = builder.aiOnnx.add([c3, c4])

    o = builder.aiOnnx.add([a1, a2])

    loss = builder.aiGraphcore.identityloss([o])

    proto = builder.getModelProto()

    anchor_names = [
        popart.reservedGradientPrefix() + i1,
        popart.reservedGradientPrefix() + i2,
        popart.reservedGradientPrefix() + i3,
        popart.reservedGradientPrefix() + i4
    ]
    dataFlow = popart.DataFlow(
        1, {
            anchor_names[0]: popart.AnchorReturnType("All"),
            anchor_names[1]: popart.AnchorReturnType("All"),
            anchor_names[2]: popart.AnchorReturnType("All"),
            anchor_names[3]: popart.AnchorReturnType("All")
        })
    optimizer = popart.ConstSGD(0.01)

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=dataFlow,
        loss=loss,
        optimizer=optimizer,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    matmul1_lhs = np.ones(matmul1_lhs_shape.shape(), dtype=np.float32)
    matmul1_rhs = np.ones(matmul1_rhs_shape.shape(), dtype=np.float32)
    matmul2_lhs = np.ones(matmul2_lhs_shape.shape(), dtype=np.float32)
    matmul2_rhs = np.ones(matmul2_rhs_shape.shape(), dtype=np.float32)

    inputs = {
        i1: matmul1_lhs,
        i2: matmul1_rhs,
        i3: matmul2_lhs,
        i4: matmul2_rhs
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)
    session.weightsFromHost()

    # Check that there are only 3 matmul convs in computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_matmuls = tu.get_compute_set_regex_count(
        r'^.+/matmulGrouped/Conv_1/Convolve$', computeSets)
    # There should be only three matmul
    assert (num_matmuls == 3)


@tu.requires_ipu_model
def test_gemm_train_cached_by_default():
    """
    In this test we check that the default behaviour is for matmul to be
    cached.
    """

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    gemmA_shape = popart.TensorInfo("FLOAT", [2, 4])
    gemmB_shape = popart.TensorInfo("FLOAT", [4, 6])
    gemmC_shape = popart.TensorInfo("FLOAT", [2, 6])

    i1 = builder.addInputTensor(gemmA_shape)
    i2 = builder.addInputTensor(gemmB_shape)
    i3 = builder.addInputTensor(gemmC_shape)

    i4 = builder.addInputTensor(gemmA_shape)
    i5 = builder.addInputTensor(gemmB_shape)
    i6 = builder.addInputTensor(gemmC_shape)

    c1 = builder.aiOnnx.gemm([i1, i2, i3])
    c2 = builder.aiOnnx.gemm([i4, i5, i6])

    o = builder.aiOnnx.add([c1, c2])

    loss = builder.aiGraphcore.identityloss([o])

    proto = builder.getModelProto()

    anchor_names = [
        popart.reservedGradientPrefix() + i1,
        popart.reservedGradientPrefix() + i2,
        popart.reservedGradientPrefix() + i3,
        popart.reservedGradientPrefix() + i4,
        popart.reservedGradientPrefix() + i5,
        popart.reservedGradientPrefix() + i6
    ]
    dataFlow = popart.DataFlow(
        1, {
            anchor_names[0]: popart.AnchorReturnType("All"),
            anchor_names[1]: popart.AnchorReturnType("All"),
            anchor_names[2]: popart.AnchorReturnType("All"),
            anchor_names[3]: popart.AnchorReturnType("All"),
            anchor_names[4]: popart.AnchorReturnType("All"),
            anchor_names[5]: popart.AnchorReturnType("All")
        })
    optimizer = popart.ConstSGD(0.01)

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=dataFlow,
        loss=loss,
        optimizer=optimizer,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    gemm1_A = np.ones(gemmA_shape.shape(), dtype=np.float32)
    gemm1_B = np.ones(gemmB_shape.shape(), dtype=np.float32)
    gemm1_C = np.ones(gemmC_shape.shape(), dtype=np.float32)

    gemm2_A = np.ones(gemmA_shape.shape(), dtype=np.float32)
    gemm2_B = np.ones(gemmB_shape.shape(), dtype=np.float32)
    gemm2_C = np.ones(gemmC_shape.shape(), dtype=np.float32)

    inputs = {
        i1: gemm1_A,
        i2: gemm1_B,
        i3: gemm1_C,
        i4: gemm2_A,
        i5: gemm2_B,
        i6: gemm2_C
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)
    session.weightsFromHost()

    # Check that there is only 2 matmul conv's computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_matmuls = tu.get_compute_set_regex_count(
        r'^.+/matmulGrouped/Conv_1/Convolve$', computeSets)
    # There should be only three matmul
    assert (num_matmuls == 3)


@tu.requires_ipu_model
def test_outlining_bca1():
    """
    In this test we check that the default behaviour is for matmul to be
    cached.
    """

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    matmul_lhs_shape = popart.TensorInfo("FLOAT", [2, 3])
    matmul_rhs_shape = popart.TensorInfo("FLOAT", [3, 4])

    i1 = builder.addInputTensor(matmul_lhs_shape)
    i2 = builder.addInputTensor(matmul_rhs_shape)
    i3 = builder.addInputTensor(matmul_lhs_shape)
    i4 = builder.addInputTensor(matmul_rhs_shape)

    c1 = builder.aiOnnx.matmul([i1, i2])
    c2 = builder.aiOnnx.matmul([i3, i4])

    a1 = builder.aiOnnx.add([c1, c2])

    c3 = builder.aiOnnx.matmul([i1, i2])
    c4 = builder.aiOnnx.matmul([i3, i4])

    a2 = builder.aiOnnx.add([c3, c4])

    o = builder.aiOnnx.add([a1, a2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    anchor_names = [o]
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    matmul1_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul1_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    matmul2_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul2_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    inputs = {
        i1: matmul1_lhs,
        i2: matmul1_rhs,
        i3: matmul2_lhs,
        i4: matmul2_rhs
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_matmuls = tu.get_compute_set_regex_count(
        r'^.+/matmulGrouped/Conv_1/Convolve$', computeSets)
    # There should be only one matmul
    assert (num_matmuls == 1)


@tu.requires_ipu_model
def test_outlining_bca2():
    """
    In this test we check that the default behaviour is for matmul to be
    cached.
    """

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    matmul_lhs_shape = popart.TensorInfo("FLOAT", [2, 3])
    matmul_rhs_shape = popart.TensorInfo("FLOAT", [3, 4])

    i1 = builder.addInputTensor(matmul_lhs_shape)
    i2 = builder.addInputTensor(matmul_rhs_shape)
    i3 = builder.addInputTensor(matmul_lhs_shape)
    i4 = builder.addInputTensor(matmul_rhs_shape)

    c1 = builder.aiOnnx.matmul([i1, i2])
    c2 = builder.aiOnnx.matmul([i3, i4])

    r1 = builder.aiOnnx.relu([c1])
    r2 = builder.aiOnnx.relu([c2])

    a1 = builder.aiOnnx.sum([r1, r2, c1, c2])

    c3 = builder.aiOnnx.matmul([i1, i2])
    c4 = builder.aiOnnx.matmul([i3, i4])

    r3 = builder.aiOnnx.relu([c3])
    r4 = builder.aiOnnx.relu([c4])

    a2 = builder.aiOnnx.sum([r3, r4, c3, c4])

    o = builder.aiOnnx.add([a1, a2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    anchor_names = [o]
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}

    session = popart.InferenceSession(
        fnModel=proto,
        dataFlow=dataFlow,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    matmul1_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul1_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    matmul2_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul2_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    inputs = {
        i1: matmul1_lhs,
        i2: matmul1_rhs,
        i3: matmul2_lhs,
        i4: matmul2_rhs
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_matmuls = tu.get_compute_set_regex_count(
        r'^.+/matmulGrouped/Conv_1/Convolve$', computeSets)
    # There should be only one matmul
    assert (num_matmuls == 1)


@tu.requires_ipu_model
def test_outlining_bca3():
    """
    In this test we check that the default behaviour is for matmul to be
    cached.
    """

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    matmul_lhs_shape = popart.TensorInfo("FLOAT", [2, 3])
    matmul_rhs_shape = popart.TensorInfo("FLOAT", [3, 4])

    i1 = builder.addInputTensor(matmul_lhs_shape)
    i2 = builder.addInputTensor(matmul_rhs_shape)
    i3 = builder.addInputTensor(matmul_lhs_shape)
    i4 = builder.addInputTensor(matmul_rhs_shape)

    c1 = builder.aiOnnx.matmul([i1, i2])
    c2 = builder.aiOnnx.matmul([i3, i4])

    a1 = builder.aiOnnx.add([c1, c2])

    c3 = builder.aiOnnx.matmul([i1, i2])
    c4 = builder.aiOnnx.matmul([i3, i4])

    a2 = builder.aiOnnx.add([c3, c4])

    o = builder.aiOnnx.add([a1, a2])

    loss = builder.aiGraphcore.identityloss([o])

    proto = builder.getModelProto()

    anchor_names = [o]
    dataFlow = popart.DataFlow(
        1, {
            o: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i1:
            popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i2:
            popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i3:
            popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i4:
            popart.AnchorReturnType("All")
        })

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}

    optimizer = popart.ConstSGD(0.01)

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=dataFlow,
        loss=loss,
        optimizer=optimizer,
        userOptions=opts,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    matmul1_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul1_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    matmul2_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul2_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    inputs = {
        i1: matmul1_lhs,
        i2: matmul1_rhs,
        i3: matmul2_lhs,
        i4: matmul2_rhs
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_matmuls = tu.get_compute_set_regex_count(
        r'^.+/matmulGrouped/Conv_1/Convolve$', computeSets)
    # There should be only 3 matmuls (fwd, bwd_lhs, bwd_rhs)
    assert (num_matmuls == 3)


@tu.requires_ipu_model
def test_outlining_bca4():
    """
    In this test we check that for the case of 2 2d tensors they can all
    use the same matmul 
    """

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    matmul_lhs_shape = popart.TensorInfo("FLOAT", [2, 2])
    matmul_rhs_shape = popart.TensorInfo("FLOAT", [2, 2])

    i1 = builder.addInputTensor(matmul_lhs_shape)
    i2 = builder.addInputTensor(matmul_rhs_shape)
    i3 = builder.addInputTensor(matmul_lhs_shape)
    i4 = builder.addInputTensor(matmul_rhs_shape)

    c1 = builder.aiOnnx.matmul([i1, i2])
    c2 = builder.aiOnnx.matmul([i3, i4])

    a1 = builder.aiOnnx.add([c1, c2])

    c3 = builder.aiOnnx.matmul([i1, i2])
    c4 = builder.aiOnnx.matmul([i3, i4])

    a2 = builder.aiOnnx.add([c3, c4])

    o = builder.aiOnnx.add([a1, a2])

    loss = builder.aiGraphcore.identityloss([o])

    proto = builder.getModelProto()

    anchor_names = [o]
    dataFlow = popart.DataFlow(
        1, {
            o: popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i1:
            popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i2:
            popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i3:
            popart.AnchorReturnType("All"),
            popart.reservedGradientPrefix() + i4:
            popart.AnchorReturnType("All")
        })

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enableFullyConnectedPass = False

    # Disabled grouped matmuls so they are all outlined as apposed to being
    # grouped into 2 groups
    opts.enableGroupedMatmuls = False

    optimizer = popart.ConstSGD(0.01)

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=dataFlow,
        loss=loss,
        optimizer=optimizer,
        userOptions=opts,
        # Enable the matmul patterns
        patterns=popart.Patterns(popart.PatternsLevel.All),
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    matmul1_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul1_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    matmul2_lhs = np.ones(matmul_lhs_shape.shape(), dtype=np.float32)
    matmul2_rhs = np.ones(matmul_rhs_shape.shape(), dtype=np.float32)

    inputs = {
        i1: matmul1_lhs,
        i2: matmul1_rhs,
        i3: matmul2_lhs,
        i4: matmul2_rhs
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    # Check that there is only one convolution computation set.
    summaryReport = session.getSummaryReport()
    computeSets = tu.get_compute_sets_from_report(summaryReport)

    num_matmuls = tu.get_compute_set_regex_count(
        r'^.+/matmulGrouped/Conv_1/Convolve$', computeSets)
    # There should be only one matmul
    assert (num_matmuls == 1)
