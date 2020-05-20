# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu
import sys
import re


def test_basic(tmpdir, capfd):
    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [3])
    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)

    a1 = builder.aiOnnx.add([i1, i2])
    p1 = builder.aiGraphcore.printtensor([a1])
    a2 = builder.aiOnnx.add([i1, p1])
    p2 = builder.aiGraphcore.printtensor([a2])

    o = p2
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.enableOutlining = False
    opts.enableOutliningCopyCostPruning = False

    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      userOptions=opts,
                                      deviceInfo=tu.create_test_device())

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {
        i1: np.array([1., 2., 3.], dtype=np.float32),
        i2: np.array([4., 5., 6.], dtype=np.float32)
    }
    stepio = popart.PyStepIO(inputs, anchors)

    capfd.readouterr()

    session.run(stepio)

    captured = capfd.readouterr()
    output = captured.err

    # patterns to match a1 and a2
    a1_pattern = f'(?:{a1})'
    a2_pattern = f'(?:{a2})'
    # pattern to match a1 or a2
    a1_or_a2 = f'(?:{a1_pattern}|{a2_pattern})'
    # pattern to match tensor values
    value_pattern = '{\d+,\d+,\d+}'
    pat = f'{a1_or_a2}: {value_pattern}'
    matches = re.findall(pat, output)

    assert len(matches) == 2
    assert matches[0] == 'Add:0: {5,7,9}'
    assert matches[1] == 'Add:0/1: {6,9,12}'


def test_train(tmpdir, capfd):
    filt_data = np.array([1., 2., 1., 2.], dtype=np.float32)
    filt_data = np.reshape(filt_data, [1, 1, 2, 2])
    input_data = np.array([1., 2., 3., 4.], dtype=np.float32)
    input_data = np.reshape(input_data, [1, 1, 2, 2])

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", input_data.shape)
    i1 = builder.addInputTensor(shape, "data")

    i2 = builder.addInitializedInputTensor(filt_data, "filter")

    # both i2 and d__i2 will be printed
    p1 = builder.aiGraphcore.printtensor([i2])

    c1 = builder.aiOnnx.conv([i1, p1],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[2, 2])

    # c1 will be printed, but d__c1 will not
    o = builder.aiGraphcore.printtensor([c1], print_gradient=0)
    l1 = builder.aiGraphcore.l1loss([o], 0.1)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.enableOutlining = False
    opts.enableOutliningCopyCostPruning = False

    session = popart.TrainingSession(fnModel=proto,
                                     dataFlow=dataFlow,
                                     userOptions=opts,
                                     optimizer=popart.ConstSGD(0.1),
                                     loss=l1,
                                     deviceInfo=tu.create_test_device())

    session.prepareDevice()

    session.weightsFromHost()

    anchors = session.initAnchorArrays()

    inputs = {i1: input_data}
    stepio = popart.PyStepIO(inputs, anchors)

    capfd.readouterr()

    session.run(stepio)

    captured = capfd.readouterr()
    output = captured.err

    # Remove ESC characters
    output = re.sub(chr(27), '', output)

    # Remove termcolor sequences
    output = re.sub('\[\d\dm', '', output)

    # Remove popart log lines
    output = re.sub('\[\d\d\d\d-\d\d-\d\d .*?\n', '', output)

    # remove all whitespace
    output = re.sub('\s+', '', output)

    pattern = 'name:{{{{float,float},{float,float}}}}'
    pattern = re.sub('name', r'[\\w:]+', pattern)
    pattern = re.sub('float', r'\\d(?:\\.\\d+)?', pattern)

    matches = re.findall(pattern, output)

    d__i2 = popart.reservedGradientPrefix() + i2

    assert len(matches) == 3
    assert matches[0] == i2 + ":{{{{1,2},{1,2}}}}"
    assert matches[1] == c1 + ":{{{{2,2},{6,4}}}}"
    assert matches[2] == d__i2 + ":{{{{0.4,0.3},{0.2,0.1}}}}"
