import sys
import os

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

# and some utility python functions.
import poponnx.writer

# Components in the core library
from poponnx_core import TensorInfo, DataFlow, NllLoss, L1Loss, SGD, ConstSGD

print("Import test passed")
