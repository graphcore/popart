import sys
from popart import *

poponnx_exception = popart_exception

print(
    "You have imported an old module 'poponnx'. Please change your code to "
    "import 'popart', poponnx will be remove at some point in the future.",
    file=sys.stderr)
