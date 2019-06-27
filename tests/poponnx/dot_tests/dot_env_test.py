import poponnx
import os
import tempfile

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_basic(tmpdir):
    def run_test(expected_dot_file_count):
        builder = poponnx.Builder()

        shape = poponnx.TensorInfo("FLOAT", [1])
        i1 = builder.addInputTensor(shape)
        o = builder.aiOnnx.identity([i1])
        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

        opts = poponnx.SessionOptions()

        with tempfile.TemporaryDirectory() as tmpdir:
            opts.logDir = tmpdir

            session = poponnx.InferenceSession(
                fnModel=proto,
                dataFeed=dataFlow,
                userOptions=opts,
                deviceInfo=tu.get_poplar_cpu_device())

            dotFiles = list(Path(tmpdir).glob('*.dot'))
            assert len(dotFiles) == expected_dot_file_count

    os.environ['POPONNX_DOT_CHECKS'] = ''
    run_test(0)

    os.environ['POPONNX_DOT_CHECKS'] = 'FWD0:FINAL'
    run_test(2)

    os.environ['POPONNX_DOT_CHECKS'] = 'FWD0:FWD1:FINAL'
    run_test(3)
