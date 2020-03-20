# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_verify_subgraph():
    builder = popart.Builder()

    # sg0
    sg0_builder = builder.createSubgraphBuilder()
    sg0_i0 = sg0_builder.addUntypedInputTensor()
    # LSTM is not outlinable
    sg0_out, _, _ = sg0_builder.aiOnnx.lstm([sg0_i0, sg0_i0, sg0_i0],
                                            3,
                                            clip=None)
    sg0_builder.addOutputTensor(sg0_out)

    # main
    i0 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2]))
    out = builder.aiGraphcore.call([i0], 1, sg0_builder)[0]

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(1, {out: popart.AnchorReturnType("ALL")}),
            deviceInfo=tu.create_test_device())
    assert (e_info.value.args[0].endswith("are not outlineable"))
