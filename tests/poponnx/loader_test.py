# Test for basic importing

import pytest


def test_import():
    # the core library
    import poponnx

    # and some utility python functions.
    import poponnx.writer

    assert ('SGD' in dir(poponnx))
