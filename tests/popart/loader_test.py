# Test for basic importing

import pytest


def test_import():
    # the core library
    import popart

    # and some utility python functions.
    import popart.writer

    assert ('SGD' in dir(popart))
