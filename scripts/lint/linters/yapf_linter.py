# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Tuple
from scripts.lint.linters.base_linter import ILinter
from scripts.lint.config import LinterConfig


class YapfLinter(ILinter):
    def __init__(self, config: LinterConfig) -> None:
        super().__init__(config)
        assert config.config_file is not None, 'YapfLinter requires the path to the config file to be specified in linter_config.json'
        self._try_import()

    def _try_import(self):
        try:
            import yapf
            self._linter_function = yapf.yapf_api.FormatCode
            self._actual_version = tuple(
                int(i) for i in yapf.__version__.split('.'))
        except (ImportError, ModuleNotFoundError):
            self._linter_function = None
            self._actual_version = None

    def is_available(self) -> bool:
        return self._linter_function is not None

    def get_version(self) -> Tuple:
        return self._actual_version

    def apply_lint_function(self, file_path: str, file_contents: str) -> str:
        self.set_linter_message("Applied yapf source code formatting.")
        linted_code, _ = self._linter_function(file_contents,
                                               filename=file_path,
                                               style_config=self.config_file)
        return linted_code

    def install_instructions(self, required_version='') -> str:
        return "pip install yapf==" + required_version
