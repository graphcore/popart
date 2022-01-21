# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
from pathlib import Path

import sys
import subprocess
import unittest


class ExamplesTester(unittest.TestCase):
    """Test case for the examples
    """

    @classmethod
    def setUpClass(cls):
        """Setup
        """
        cls.cwd = os.getcwd()
        cls.default_timeout = 600.0
        cls.base_path = Path(os.path.dirname(__file__)).parent

    def run_command(self, command, working_dir, timeout_secs):
        completed = subprocess.run(args=command.split(),
                                   cwd=working_dir,
                                   shell=False,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   timeout=timeout_secs)
        combined_output = str(completed.stdout, 'utf-8')
        try:
            completed.check_returncode()
            return_code_ok = True
        except subprocess.CalledProcessError:
            return_code_ok = False

        if not return_code_ok:
            self.fail(f"""The following command failed: {command}\n
                Working path: {self.cwd}\n
                Output of failed command:\n{combined_output}""")

    def run_python(self, filename, dir_, timeout_secs=600.0):
        py_exec = sys.executable
        filedir = os.path.join(dir_, filename)
        cmd = f"{py_exec} {filedir}"
        self.run_command(cmd, self.cwd, timeout_secs)


class TestPythonDocExamples(ExamplesTester):
    """Test simple running of the examples included in the docs"""

    def test_documentation_import_graph(self):
        """"Test the import graph example"""
        filename = "importing_graphs.py"
        self.run_python(filename, self.base_path)

    def test_documentation_import_session(self):
        """Test the import session example"""
        filename = "importing_session.py"
        self.run_python(filename, self.base_path)

    def test_documentation_execute_graph(self):
        """Test the executing imported model example"""
        filename = "executing_imported_model.py"
        self.run_python(filename, self.base_path)

    def test_documentation_simple_addition(self):
        """Test the simple addition example"""
        filename = "simple_addition.py"
        self.run_python(filename, self.base_path)

    def test_documentation_weights(self):
        """Test the weights example"""
        filename = "weights.py"
        self.run_python(filename, self.base_path)

    def test_documentation_popart_ir_addition(self):
        """Test the popart.ir simple addition example"""
        filename = "simple_addition_popart_ir.py"
        self.run_python(filename, self.base_path)

    def test_documentation_popart_ir_addition_variable(self):
        """Test the popart.ir simple addition example"""
        filename = "tensor_addition_popart_ir.py"
        self.run_python(filename, self.base_path)

    def test_documentation_popart_ir_basic_subgraph(self):
        """Test the popart.ir basic subgraph example"""
        filename = "basic_subgraph_popart_ir.py"
        self.run_python(filename, self.base_path)

    def test_documentation_popart_ir_create_multi_subgraph(self):
        """Test the popart.ir create multiple subgraph example"""
        filename = "create_multi_subgraphs_from_same_func_popart_ir.py"
        self.run_python(filename, self.base_path)

    def test_documentation_popart_ir_multi_callsites_subgraph_input(self):
        """Test the popart.ir create multiple callsites for a subgraph input example"""
        filename = "multi_call_subgraph_input_popart_ir.py"
        self.run_python(filename, self.base_path)

    def test_documentation_popart_ir_call_with_info(self):
        """Test the popart.ir call_with_info example"""
        filename = "call_with_info_popart_ir.py"
        self.run_python(filename, self.base_path)

    def test_documentation_popart_ir_repeat_0(self):
        """Test the popart.ir basic repeat example"""
        filename = "repeat_subgraph_popart_ir_0.py"
        self.run_python(filename, self.base_path)

    def test_documentation_popart_ir_repeat_1(self):
        """Test the popart.ir subgraph in parent in repeat example"""
        filename = "repeat_subgraph_popart_ir_1.py"
        self.run_python(filename, self.base_path)


class TestPopartCustomOperatorCube(ExamplesTester):
    """Tests for example of Popart cube custom operator"""

    def setUp(self):
        self.build_dir = os.path.join(self.base_path, "custom_op")
        self.run_command("make clean", self.build_dir, self.default_timeout)
        self.run_command("make", self.build_dir, 600.0)

    def test_run_cube_custom_op(self):
        filename = "custom_op.py"
        self.run_python(filename, self.base_path, 600.0)
