# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
import shutil
import subprocess

from lint.config import LinterConfig
from lint.linters import ILinter
from util import bash, get_project_source_dir

__all__ = ["ClangFormatLinter"]


class ClangFormatLinter(ILinter):
    def __init__(self, config: LinterConfig):
        super().__init__(config)

    def apply_lint_function(self, filename, file_contents):
        self.set_linter_message("Applied clang-format style changes.")
        # We don't need to supply a path to the config file because when
        # we invoke clang-format, we provide cwd as the root popart dir.
        # This directory contains the clang-format config, so clang-format
        # will search the directory it was invoked from for this file.
        process = subprocess.run(
            ["clang-format", "--style=file", f"--assume-filename={filename}"],
            cwd=get_project_source_dir(),
            input=file_contents,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        return process.stdout

    def get_version(self):
        if self.is_available():
            version_str = bash(["clang-format", "--version"], log=False)
            m = re.search(
                r"version (?P<version>\d+\.\d+\.\d+) \([\w\/:.-]+ (?P<hash>[a-z0-9]+)\)$",
                version_str.strip())
            if m:
                version = (*(int(i) for i in m.group('version').split('.')),
                           m.group('hash'))
                return version

    def is_available(self):
        return shutil.which("clang-format") is not None

    def install_instructions(self, required_version=''):
        return f"pip install clang-format=={required_version}"
