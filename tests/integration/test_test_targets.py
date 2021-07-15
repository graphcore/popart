import os
import test_util as tu
import pytest


@tu.requires_ipu
def test_requires_ipu():
    assert os.environ['TEST_TARGET'] == 'Hw'


@tu.requires_ipu_model
def test_requires_ipu_model():
    assert os.environ['TEST_TARGET'] == 'IpuModel'


def test_should_be_empty():
    assert os.environ['TEST_TARGET'] == ''


# This test fails allowing the test after to check that
# TEST_TARGET is cleaned up in the case of a failing test.
@pytest.mark.xfail
@tu.requires_ipu
def test_fail_on_purpose():
    assert False


def test_should_be_empty():
    assert os.environ['TEST_TARGET'] == ''
