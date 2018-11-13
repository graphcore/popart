import os, sys

# Add the DSO library path
lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")
lp = os.path.abspath(lp)
sys.path.insert(0, lp)

# Import all symbols into our namespace
from poponnx_core import *
from poponnx.session import Session
