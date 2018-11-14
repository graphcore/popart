# Test for basic importing

import pytest

# the core library
import poponnx


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
