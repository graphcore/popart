import numpy as np
import poponnx
import pytest
import test_util as tu


def test_create_empty_options():

    opts = poponnx.SessionOptions()
    assert (opts is not None)
    assert (opts.exportDot == False)
    assert (len(opts.engineOptions) == 0)
    assert (len(opts.convolutionOptions) == 0)
    assert (len(opts.reportOptions) == 0)


def test_set_exportDot_flag():

    opts = poponnx.SessionOptions()
    opts.exportDot = True

    assert (opts.exportDot == True)


def test_set_enableConvolutionGraphCaching_flag():

    opts = poponnx.SessionOptions()
    assert (opts.enableConvolutionGraphCaching == True)

    opts.enableConvolutionGraphCaching = False
    assert (opts.enableConvolutionGraphCaching == False)


def test_set_engineOptions():

    opts = poponnx.SessionOptions()
    opts.engineOptions = {'option': 'value'}

    assert (len(opts.engineOptions) == 1)
    assert (len(opts.convolutionOptions) == 0)
    assert (len(opts.reportOptions) == 0)
    assert (opts.engineOptions['option'] == 'value')


def test_set_convolutionOptions():

    opts = poponnx.SessionOptions()
    opts.convolutionOptions = {'option': 'value'}

    assert (len(opts.engineOptions) == 0)
    assert (len(opts.convolutionOptions) == 1)
    assert (len(opts.reportOptions) == 0)
    assert (opts.convolutionOptions['option'] == 'value')


def test_set_reportOptions():

    opts = poponnx.SessionOptions()
    opts.reportOptions = {'option': 'value'}

    assert (len(opts.engineOptions) == 0)
    assert (len(opts.convolutionOptions) == 0)
    assert (len(opts.reportOptions) == 1)
    assert (opts.reportOptions['option'] == 'value')


def test_engine_options_passed_to_engine(tmpdir):
    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [1, 2, 32, 32])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)

    o = builder.add([i1, i2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    opts = poponnx.SessionOptions()
    opts.engineOptions = {'option': 'value'}
    opts.logging = {'all': 'DEBUG'}

    session = poponnx.Session(
        fnModel=proto,
        dataFeed=dataFlow,
        outputdir=str(tmpdir),
        userOptions=opts)

    session.setDevice(tu.get_poplar_cpu_device())
    session.initAnchorArrays()

    with pytest.raises(poponnx.poplar_exception) as e_info:
        session.prepareDevice()

    assert (e_info.value.args[0].endswith("Unrecognised option 'option'"))


def test_convolution_options(tmpdir):

    builder = poponnx.Builder()

    data_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    filt_shape = poponnx.TensorInfo("FLOAT", [3, 2, 3, 3])

    i1 = builder.addInputTensor(data_shape)
    i2 = builder.addInputTensor(filt_shape)
    o = builder.convolution([i1, i2], [1, 1], [1, 1, 1, 1], [1, 1], 1)
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, 1, [o], poponnx.AnchorReturnType.ALL)

    opts = poponnx.SessionOptions()
    opts.convolutionOptions = {'startTileMultiplier': '3'}
    opts.logging = {'all': 'DEBUG'}

    session = poponnx.Session(
        fnModel=proto,
        dataFeed=dataFlow,
        outputdir=str(tmpdir),
        userOptions=opts)

    session.setDevice(tu.get_poplar_cpu_device())
    anchors = session.initAnchorArrays()

    with pytest.raises(poponnx.poplibs_exception) as e_info:
        session.prepareDevice()

    assert (e_info.value.args[0].endswith(
        "Must start distributing convolutions on an even tile."))
