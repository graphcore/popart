import sys
import os

import pytest

# We test if we can load poponnx. this requires that the python
# library and its dependencies are found.

## Search for poponnx .so/.dylib when importing
testdir = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.join(testdir, "../../lib")
sys.path.append(libpath)

if sys.platform != "darwin":
    # So python finds libwillow.so when importing poponnx
    # (without having to export LD_LIBRARY_PATH)
    import ctypes
    ctypes.cdll.LoadLibrary(os.path.join(libpath, "libpoponnx.so"))

# Required for torchwriter if install/willow/python not in PYTHONPATH env var
pypath = os.path.join(testdir, "../../python")
sys.path.append(pypath)

# the core library
import poponnx

# Components in the core library
from poponnx_core import Builder, TensorInfo

def test_basic():

    builder = Builder()

    i1 = builder.addInputTensor(TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(TensorInfo("FLOAT", [1, 2, 32, 32]))

    o = builder.add(i1, i2)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert(len(proto) > 0)
    assert(len(i1) > 0)
    assert(len(i2) > 0)
    assert(len(o) > 0)
    assert(i1 != i2)
    assert(i2 != o)
