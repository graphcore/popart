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

    def test_documentation_popxl_addition(self):
        """Test the popxl simple addition example"""
        filename = "simple_addition.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_addition_variable(self):
        """Test the popxl simple addition example"""
        filename = "tensor_addition.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_basic_subgraph(self):
        """Test the popxl basic subgraph example"""
        filename = "basic_graph.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_create_multi_subgraph(self):
        """Test the popxl create multiple subgraph example"""
        filename = "create_multi_graphs_from_same_func.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_multi_callsites_graph_input(self):
        """Test the popxl create multiple callsites for a subgraph input example"""
        filename = "multi_call_graph_input.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_call_with_info(self):
        """Test the popxl call_with_info example"""
        filename = "call_with_info.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_repeat_0(self):
        """Test the popxl basic repeat example"""
        filename = "repeat_graph_0.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_repeat_1(self):
        """Test the popxl subgraph in parent in repeat example"""
        filename = "repeat_graph_1.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_repeat_2(self):
        """Test the popxl subgraph in parent in repeat example"""
        filename = "repeat_graph_2.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popart_ir_get_set_tensors(self):
        """Test the popxl getting / setting tensor data example"""
        filename = "tensor_get_write.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_autodiff(self):
        """Test the popxl autodiff op"""
        filename = "autodiff.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_in_sequence(self):
        """Test the popxl in sequence context manager"""
        filename = "in_sequence.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_remote_var(self):
        """Test the popxl remote variable"""
        filename = "remote_variable.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_remote_rts_var(self):
        """Test the popxl remote rts variable"""
        filename = "remote_rts_var.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_rts_var(self):
        """Test the popxl rts variable"""
        filename = "rts_var.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_mnist(self):
        """Test the popxl basic mnist example"""
        filename = "mnist.py"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_mnist_replication_train(self):
        """Test the popxl mnist with replication example"""
        filename = "mnist_rts.py --replication-factor 2"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_mnist_rts_train(self):
        """Test the popxl mnist with RTS example"""
        filename = "mnist_rts.py --replication-factor 2 --rts"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)

    def test_documentation_popxl_mnist_rts_train_test(self):
        """Test the popxl mnist with RTS example"""
        filename = "mnist_rts.py --replication-factor 2 --rts --test"
        self.run_python(filename,
                        file_dir=working_dir,
                        working_dir=working_dir)
