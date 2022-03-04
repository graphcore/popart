# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart

# importing test_session requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import PopartTestSession
import test_util as tu


def test_for_warning(capfd):
    data = np.zeros(2, dtype=np.float32)

    def init_builder(builder):
        x = builder.aiOnnx.constant(data)
        x = builder.aiOnnx.relu([x])
        x = builder.aiOnnx.identity([x])

        builder.addOutputTensor(x)

        return [x]

    popart.getLogger().setLevel("TRACE")

    session = PopartTestSession()
    with tu.create_test_device() as device:
        session.prepare(init_builder, device=device)

    _, err = capfd.readouterr()
    print(err)
    err = err.splitlines()

    warns = [i for i in err if 'No ConstExpr implementation of ' in i]
    assert len(warns) > 1
    assert 'Relu' in warns[0]
