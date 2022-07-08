# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import os
import sys
import subprocess
import unittest


class ExamplesTester(unittest.TestCase):
    """Test case for the examples"""

    @classmethod
    def setUpClass(cls):
        """Setup"""
        cls.cwd = os.getcwd()
        cls.default_timeout = 600.0

    def run_command(self, command, working_dir, timeout_secs):
        completed = subprocess.run(
            args=command.split(),
            cwd=working_dir,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_secs,
        )
        combined_output = str(completed.stdout, "utf-8")
        try:
            completed.check_returncode()
            return_code_ok = True
        except subprocess.CalledProcessError:
            return_code_ok = False

        if not return_code_ok:
            self.fail(
                f"""The following command failed: {command}\n
                Working path: {self.cwd}\n
                Output of failed command:\n{combined_output}"""
            )

    def run_python(
        self, filename, file_dir, working_dir, timeout_secs=600.0, *args, **kwargs
    ):
        py_exec = sys.executable
        file_path = os.path.join(file_dir, filename)
        cmd = f"{py_exec} {file_path}{' '.join([str(a) for a in args])}"
        for k, v in kwargs.items():
            k_str = f" --{k} {str(v)}"
            cmd += k_str
        self.run_command(cmd, working_dir, timeout_secs)
