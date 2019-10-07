import numpy as np
import os

# importing test_session requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import TestSession


def test_for_warning(capfd):
    data = np.zeros(2, dtype=np.float32)

    def init_builder(builder):
        x = builder.aiOnnx.constant(data)
        x = builder.aiOnnx.relu([x])
        x = builder.aiOnnx.identity([x])

        builder.addOutputTensor(x)

        return [x]

    os.environ['POPART_LOG_LEVEL'] = 'TRACE'
    os.environ['POPART_LOG_FORMAT'] = '%v'

    session = TestSession()
    session.prepare(init_builder)

    _, err = capfd.readouterr()
    print(err)
    err = err.splitlines()

    warns = [i for i in err if i.startswith('No ConstExpr implementation of ')]
    assert len(warns) > 1
    assert 'Relu' in warns[0]
