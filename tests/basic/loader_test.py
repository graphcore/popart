import sys
import os

# We test if we can load pywillow. this requires that the python
# library and its dependencies are found.

## Search for pywillow .so/.dylib when importing
testdir = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.join(testdir, "../../lib")
sys.path.append(libpath)

if sys.platform != "darwin":
    # So python finds libwillow.so when importing pywillow
    # (without having to export LD_LIBRARY_PATH)
    import ctypes
    ctypes.cdll.LoadLibrary(os.path.join(libpath, "libwillow.so"))

# Required for torchwriter if install/willow/python not in PYTHONPATH env var
pypath = os.path.join(testdir, "../../python")
sys.path.append(pypath)

# the core library, which wraps C++ willow,
import pywillow

# and some utility python functions.
import writer

print("Import test passed")
