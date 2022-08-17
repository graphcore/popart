# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path

import sys

# In the source dir the examples_tester module lives in docs/shared/files/tests.
# We add this to the system path so this file can be ran from the source dir.
dir_docs = Path(__file__).parent.parent.parent.parent.resolve()
dir_shared_tests = dir_docs.joinpath("shared", "files", "tests")
sys.path.append(str(dir_shared_tests))

from examples_tester import ExamplesTester

working_dir = Path(os.path.dirname(__file__)).parent


class TestPythonDocExamples(ExamplesTester):
    """Test simple running of the examples included in the docs"""

    def test_documentation_custom_op(self):
        """Test the weights example"""
        filename = "run_leaky_relu.py"
        self.run_python(
            filename, file_dir=working_dir, working_dir=working_dir, input_data=0.1
        )
