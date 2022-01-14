#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import subprocess
from pathlib import Path
import yaml


class VersionChecker:
    """Checks that the installed binaries needed for linting is at the right version."""

    def __init__(self) -> None:
        """Set up class and populate members by calling check_version.
        """
        version_file = Path(__file__).parent.joinpath("versions.yaml")

        with version_file.open("r") as file:
            self.version_list = yaml.load(file, Loader=yaml.SafeLoader)

        # Add the clang versions
        self.version_list["clang"] = self.version_list["llvm"]
        self.version_list["clang-tidy"] = self.version_list["llvm"]

        # llvm is not a linter, but the project behind clang
        self.version_list.pop("llvm")

        self.missing_errors = dict()
        self.version_errors = dict()
        self.linter_with_errors = set()

        self.cur_dir = Path(__file__).parent

        self.check_versions()

    def check_versions(self) -> None:
        """Make a subprocess.call to check whether the --version is correct.
        """
        for linter, version in self.version_list.items():
            try:
                result = subprocess.run((linter, "--version"),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        check=False)
                std_out = result.stdout.decode("utf-8")
                if not version in std_out:
                    self.version_errors[
                        linter] = f"{linter} does not have the correct version\n{linter} --version returned\n{std_out}\n"
                    self.linter_with_errors.add(linter)
            except FileNotFoundError:
                self.missing_errors[linter] = f"{linter} was not found."
                self.linter_with_errors.add(linter)

    def print_errors(self) -> None:
        """Print all errors.
        """
        if len(self.missing_errors) != 0:
            self.print_missing_errors()
            print()
        if len(self.version_errors) != 0:
            self.print_version_errors()
            print()
        if len(self.missing_errors) != 0 or len(self.version_errors) != 0:
            repo_dir = Path(__file__).parents[3]
            relative_path = self.cur_dir.joinpath(
                'install_linters.py').relative_to(repo_dir)
            module_path = str(relative_path.with_suffix("")).replace("/", ".")
            print("You can install the correct linter version using "
                  f"python3 -m {module_path}.")

    def print_missing_errors(self) -> None:
        """Print the linters not found.
        """
        for _, error in self.missing_errors.items():
            print(error)

    def print_version_errors(self) -> None:
        """Print the linters which did not coincide with the expected version.
        """
        for _, error in self.version_errors.items():
            print(error)

        if len(self.version_errors) != 0:
            print(
                f"See {self.cur_dir.joinpath('versions.yaml')} for version specification."
            )


def main() -> int:
    version_checker = VersionChecker()
    if len(version_checker.linter_with_errors) == 0:
        return 0
    else:
        version_checker.print_errors()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
