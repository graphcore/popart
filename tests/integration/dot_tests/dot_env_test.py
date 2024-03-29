# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import tempfile

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_basic(monkeypatch):
    def run_test(expected_dot_file_count):
        builder = popart.Builder()

        shape = popart.TensorInfo("FLOAT", [1])
        i1 = builder.addInputTensor(shape)
        o = builder.aiOnnx.identity([i1])
        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()

        with tempfile.TemporaryDirectory() as tmpdir:
            opts.logDir = tmpdir

            with tu.create_test_device() as device:
                _ = popart.InferenceSession(
                    fnModel=proto,
                    dataFlow=dataFlow,
                    userOptions=opts,
                    deviceInfo=device,
                )

                dotFiles = list(Path(tmpdir).glob("*.dot"))
                assert len(dotFiles) == expected_dot_file_count

    monkeypatch.setenv("POPART_DOT_CHECKS", "")
    run_test(0)

    monkeypatch.setenv("POPART_DOT_CHECKS", "FWD0:FINAL")
    run_test(2)

    monkeypatch.setenv("POPART_DOT_CHECKS", "FWD0:FWD1:FINAL")
    run_test(3)
