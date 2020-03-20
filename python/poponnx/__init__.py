# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import sys
import popart
from popart import *

poponnx_exception = popart_exception

__version__ = popart.__version__

print(
    "You have imported an old module 'poponnx'. Please change your code to "
    "import 'popart', poponnx will be remove at some point in the future.",
    file=sys.stderr)
