# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from scripts.lint.config import LinterConfig

__all__ = ["ILinter"]


class ILinter(ABC):
    """Base class for all the linters."""

    def __init__(self, config: LinterConfig) -> None:
        self.name = config.name
        self.include = self._convert_to_string_if_regex_list(config.include)
        self.exclude = self._convert_to_string_if_regex_list(config.exclude)
        self.config_file = config.config_file
        self._linter_message = 'Lint error.'

    def _convert_to_string_if_regex_list(self,
                                         regex: Union[str, List[str]]) -> str:
        if regex is not None:
            if type(regex) is list:
                if not all(type(s) is str for s in regex):
                    raise ValueError(
                        "Include or exclude fields must contain string regular expressions."
                    )
                return r"|".join(regex)
            elif type(regex) is str:
                return regex
            else:
                raise ValueError(
                    "Include or exclude fields must be a list of or a single regular expression."
                )

    def apply(self, filename: str, file_contents: str) -> str:
        """Apply the linter to the given file, if its available and applicable.

        If you are implementing your own linter, you should NOT override this method,
        and override `apply_lint_function` instead.
        """
        if self.is_available() and self.is_applicable(filename):
            return self.apply_lint_function(filename, file_contents)
        else:
            return file_contents

    def set_linter_message(self, message: str) -> None:
        """"Set the message describing and/or explaining the changes applied by this linter."""
        self._linter_message = message

    def get_linter_message(self) -> str:
        return self._linter_message

    @abstractmethod
    def apply_lint_function(self, file_path: str, file_contents: str) -> str:
        """Run the linter on a string representing the contents of a source file.

        This method must only return a string containing the file with any changes.
        If no changes were made, return the content of the original file.

        Optionally pass the path to the file if the linter requires knowing
        file metadata or only operates in-place.

        If you are implementing a new linter, this is the method that is most
        important to override in your linter, as it is what does the 'work'
        of linting.
        """

    @abstractmethod
    def get_version(self) -> Tuple:
        """
        Return the version of the linter package or binary.

        The version will be returned as a tuple, leading with major version numbers.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """"Check if the linter is installed."""

    @abstractmethod
    def install_instructions(self, required_version='') -> str:
        """"Return a string describing how the linter should be installed."""

    def is_applicable(self, filename: str) -> bool:
        """Return true if this linter is applicable to filename, false otherwise."""
        # If excludes are not defined then matches_exclude is vacuously False
        if self.exclude is not None:
            matches_exclude = re.search(self.exclude, filename)
        else:
            matches_exclude = False
        # If includes are not defined then matches_include is vacuously True
        if self.include is not None:
            matches_include = re.search(self.include, filename)
        else:
            matches_include = True
        return matches_include and not matches_exclude
