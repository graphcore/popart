# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
import sys

PYTHON_EXAMPLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "docs", "user_guide",
                 "python_examples"))

sys.path.insert(0, PYTHON_EXAMPLE_PATH)


def test_importing_graphs():
    import importing_graphs
    print("importing_graphs.py example succeeded")


def test_importing_ession():
    import importing_session
    print("importing_session.py example succeeded")
