# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from lint.config import LinterConfig
from lint.linters import ILinter


class LinterUnderTest(ILinter):
    def __init__(self, config: LinterConfig = None) -> None:
        super().__init__(config)

    def apply_lint_function(self, file_path, file_contents):
        pass

    def get_version(self):
        return (0, 0, 0)

    def is_available(self):
        return True

    def install_instructions(self, required_version=''):
        return ''


class UnavailableLinter(ILinter):
    def __init__(self, config: LinterConfig = None) -> None:
        super().__init__(config)

    def apply_lint_function(self, file_path, file_contents):
        pass

    def get_version(self):
        return (0, 0, 0)

    def is_available(self):
        return False

    def install_instructions(self, required_version=''):
        return 'UnavailableLinterInstallMessage'


class DeleteLinter(ILinter):
    def __init__(self, config: LinterConfig = None) -> None:
        super().__init__(config)

    def apply_lint_function(self, file_path, file_contents):
        self.set_linter_message("Wiped the file.")
        return ''

    def get_version(self):
        return (0, 0, 0)

    def is_available(self):
        return True

    def install_instructions(self, required_version=''):
        return ''


class _CommonTestMethods:
    def is_available(self):
        return True

    def is_applicable(self, filename: str):
        return True

    def get_version(self):
        return (0, 0, 0)

    def install_instructions(self):
        return ''


class FirstLinter(_CommonTestMethods):
    name = "FirstLinter"

    def __init__(self, config: LinterConfig = None) -> None:
        pass

    def apply(self, file_to_lint: str, file_contents: str):
        return file_contents + "FirstLinter\n"

    def get_linter_message(self) -> str:
        return "A"


class SecondLinter(_CommonTestMethods):
    name = "Secondlinter"

    def __init__(self, config: LinterConfig = None) -> None:
        pass

    def apply(self, file_to_lint: str, file_contents: str):
        return file_contents + "SecondLinter\n"

    def get_linter_message(self) -> str:
        return "B"


class ThirdLinter(_CommonTestMethods):
    name = "ThirdLinter"

    def __init__(self, config: LinterConfig = None) -> None:
        pass

    def apply(self, file_to_lint: str, file_contents: str):
        return file_contents + "ThirdLinter\n"

    def get_linter_message(self) -> str:
        return "C"


class DoNothingLinter(_CommonTestMethods):
    name = "DoNothingLinter"

    def __init__(self, config: LinterConfig = None) -> None:
        pass

    def apply(self, file_to_lint: str, file_contents: str):
        return file_contents

    def get_linter_message(self) -> str:
        return "Did nothing."
