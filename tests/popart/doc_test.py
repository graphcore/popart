import os
import sys

PYTHON_EXAMPLE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "docs", "user_guide",
                 "python_examples"))

sys.path.insert(0, PYTHON_EXAMPLE_PATH)


def test_importing():
    import importing
