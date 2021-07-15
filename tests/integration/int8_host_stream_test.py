# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from collections import namedtuple
import numpy as np
import pytest
import popart

import test_util as tu

_DataType = namedtuple('_DataType', ['builder_type', 'np_type'])
_INT8 = _DataType('INT8', np.int8)
_UINT8 = _DataType('UINT8', np.uint8)


@pytest.mark.parametrize("cast_type", [_INT8, _UINT8])
def test_int8_hd_stream_then_cast_then_op_then_cast_then_int8_dh_stream(
        cast_type):
    """
    Test can stream an int8 input to device, and an int8 output back to host.

    -> stream int8 host to device
    -> cast to float16
    -> scale by 5
    -> cast to int8
    -> stream int8 device to host
    Check returned host tensor is correct.
    """
    scale_factor = 5

    #### Build model ###
    builder = popart.Builder()

    in0_host = np.array([1, 3], dtype=cast_type.np_type)
    shape = list(in0_host.shape)

    in0 = builder.addInputTensor(cast_type.builder_type, shape)
    t1 = builder.aiOnnx.cast([in0], "FLOAT16")

    t2 = builder.aiGraphcore.scale([t1], scale_factor)

    out = builder.aiOnnx.cast([t2], cast_type.builder_type)

    ### Create session and run program ###
    s = popart.InferenceSession(fnModel=builder.getModelProto(),
                                dataFlow=popart.DataFlow(1, [out]),
                                deviceInfo=tu.create_test_device())
    s.prepareDevice()

    anchors = s.initAnchorArrays()

    inputs = {
        in0: in0_host,
    }
    stepio = popart.PyStepIO(inputs, anchors)

    s.run(stepio)

    ### Numerically compare scaled tensor to expected value ###

    expected = in0_host * scale_factor

    assert (np.array_equal(anchors[out], expected))


@pytest.mark.parametrize("cast_type", [_INT8, _UINT8])
def test_fail_stream_int8_no_cast_then_op(cast_type):
    """
    Test using the int8 tensors in an add op before casting fails.

    -> Stream int8 tensors to device
    -> Add them
    Check fails in poplibs (in session.prepareDevice)
    """

    builder = popart.Builder()

    tinfo = popart.TensorInfo(cast_type.builder_type, [2])
    i1 = builder.addInputTensor(tinfo)
    i2 = builder.addInputTensor(tinfo)

    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()
    dataFlow = popart.DataFlow(
        1, {
            i1: popart.AnchorReturnType("Final"),
            i2: popart.AnchorReturnType("Final"),
            o: popart.AnchorReturnType("Final")
        })
    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device())

    with pytest.raises(popart.poplar_exception):
        session.prepareDevice()


@pytest.mark.parametrize("cast_type", [_INT8, _UINT8])
def test_pipelining_recomp(cast_type):
    """
    We test all the following conditions for int8 tensors:
      1. When recomputing a pipeline stage, the input to that stage is stashed.
      2. If tensor consumed by multiple pipeline stages, it is stashed.
      3. If tensor consumed by later pipeline stage on different virtual graph,
         it will be IpuCopy'd.
    
    We simply test no error is thrown during compilation or runtime. Other tests
    make sufficient numerical checks.
    """

    bps = 5

    dshape = [1, 2, 4, 4]
    w_data = np.random.rand(*dshape).astype(np.float16)
    x_i8_data = np.random.rand(bps, *dshape).astype(cast_type.np_type)

    #### Build model ###

    builder = popart.Builder()

    x_i8 = builder.addInputTensor(cast_type.builder_type, dshape, "x")
    w = builder.addInitializedInputTensor(w_data)

    with builder.virtualGraph(0), builder.pipelineStage(0):
        # (1) Stashed due to recomputation.
        x = builder.aiOnnx.cast([x_i8], "FLOAT16")
        y = builder.aiOnnx.mul([x, w])
        # (2, 3) IpuCopy'd to stage 1, and stashed as also consumed by stage 2.
        y = builder.aiOnnx.cast([y], cast_type.builder_type)

    with builder.virtualGraph(1), builder.pipelineStage(1):
        y = builder.aiOnnx.cast([y], "FLOAT16")
        y = builder.aiOnnx.sqrt([y])

    with builder.virtualGraph(0), builder.pipelineStage(2):
        y = builder.aiOnnx.mul([w, y])
        loss = builder.aiGraphcore.identityloss([y])
        loss_i8 = builder.aiOnnx.cast([loss], cast_type.builder_type)
        builder.addOutputTensor(loss_i8)

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.enablePipelining = True
    opts.enableGradientAccumulation = True
    opts.accumulationFactor = bps
    opts.accumulationAndReplicationReductionType = popart.ReductionType.Mean
    opts.decomposeGradSum = True
    opts.autoRecomputation = popart.RecomputationType.Pipeline
    opts.explicitRecomputation = False

    ### Create session and run program ###

    session = popart.TrainingSession(
        deviceInfo=tu.create_test_device(numIpus=2),
        dataFlow=popart.DataFlow(1, [loss, w],
                                 popart.AnchorReturnType("Final")),
        fnModel=builder.getModelProto(),
        loss=loss,
        optimizer=popart.ConstSGD(0.1),
        userOptions=opts)

    session.prepareDevice()

    session.weightsFromHost()
    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({x_i8: x_i8_data}, anchors)
    session.run(stepio)
