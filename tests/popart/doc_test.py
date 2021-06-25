# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
from pathlib import Path
import os
import sys

PYTHON_EXAMPLE_PATH = (Path(__file__).parent.parent.parent /
                       'docs/user_guide/files').resolve()
assert PYTHON_EXAMPLE_PATH.exists()

sys.path.insert(0, str(PYTHON_EXAMPLE_PATH))


def test_importing_graphs():
    import importing_graphs
    print("importing_graphs.py example succeeded")


def test_importing_ession():
    import importing_session
    print("importing_session.py example succeeded")
