# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

    def test_documentation_import_graph(self):
        """"Test the import graph example"""
        filename = "importing_graphs.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_import_session(self):
        """Test the import session example"""
        filename = "importing_session.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_execute_graph(self):
        """Test the executing imported model example"""
        filename = "executing_imported_model.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_simple_addition(self):
        """Test the simple addition example"""
        filename = "simple_addition.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_weights(self):
        """Test the weights example"""
        filename = "weights.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_custom_op(self):
        """Test the weights example"""
        filename = "run_leaky_relu.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir,
                        input_data=0.1)

    def test_documentation_replication(self):
        """Test replication"""
        filename = "replication_popart.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)


class TestPopartCustomOperatorCube(ExamplesTester):
    """Tests for example of Popart cube custom operator"""

    def setUp(self):
        custom_op_dir = os.path.join(working_dir, "custom_op")
        self.run_command("make clean",
                         working_dir=custom_op_dir,
                         timeout_secs=self.default_timeout)
        self.run_command("make", working_dir=custom_op_dir, timeout_secs=600.0)

    def test_run_cube_custom_op(self):
        filename = "custom_op.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir,
                        timeout_secs=600.0)
