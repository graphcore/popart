# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import inspect
import subprocess
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

# See
# https://docs.pytest.org/en/latest/how-to/monkeypatch.html for
# details on monkey patching
from pytest import MonkeyPatch
from scripts.lint.linters.check_versions import VersionChecker


class MonkeySubprocessRun:
    """Class used to patch the subprocess.run function."""

    def __init__(self, std_out_list: List[str], raise_error: bool = False) -> None:
        """Set the contents of the standard output, and if an error should be raised.

        Args:
            std_out_list (List[str]): List of standard outputs to emit
            raise_error (bool, optional): Whether or not an error should be raised.
                Defaults to False.
        """
        self.std_out_list = std_out_list
        self.raise_error = raise_error

    def __call__(self, *_, **__) -> "MonkeyCompletedProcess":
        """Return a MonkeyCompletedProcess containing the member data of this class.

        Returns:
            MonkeyCompletedProcess: Object patching CompletedProcess
        """
        return MonkeyCompletedProcess(self.std_out_list, self.raise_error)


class MonkeyCompletedProcess:
    """Class used to patch the CompletedProcess class."""

    # As the calls to the patches calls the constructor several times, we make
    # a static variable to keep track of number of calls to stdout
    # As several instances may use the same object we will use the current
    # stack as the key
    calls_to_stdout = dict()

    def __init__(self, std_out_list: List[str], raise_error: bool = False) -> None:
        """Set the contents of the standard output, and if an error should be raised.

        Args:
            std_out_list (List[str]): List of standard outputs to emit
            raise_error (bool, optional): Whether or not an error should be raised.
                Defaults to False.
        """
        self.std_out_list = std_out_list
        self.raise_error = raise_error
        self.current_stack = "/".join([stack.function for stack in inspect.stack()])
        if self.current_stack not in MonkeyCompletedProcess.calls_to_stdout:
            MonkeyCompletedProcess.calls_to_stdout[self.current_stack] = -1

    @property
    def stdout(self) -> bytes:
        """Get stdout.

        Raises:
            FileNotFoundError: If self.raise_error is true

        Returns:
            bytes: The next element in the std_out_list
        """
        if self.raise_error:
            raise FileNotFoundError

        MonkeyCompletedProcess.calls_to_stdout[self.current_stack] += 1
        return self.std_out_list[
            MonkeyCompletedProcess.calls_to_stdout[self.current_stack]
        ].encode("utf-8")


@pytest.fixture(scope="function", name="version_dict")
def fixture_version_dict() -> Dict[str, str]:
    """Return the version dict.

    Returns:
        Dict[str, str]: The dict containing a pair of linter name and version
    """
    version_file = (
        Path(__file__)
        .parents[2]
        .joinpath("scripts", "lint", "install", "versions.yaml")
    )
    with version_file.open("r") as file:
        version_dict = yaml.load(file, Loader=yaml.SafeLoader)
    version_dict["clang"] = version_dict["llvm"]
    version_dict["clang-tidy"] = version_dict["llvm"]
    version_dict.pop("llvm")

    return version_dict


def test_version_checker_success(
    version_dict: Dict[str, str], monkeypatch: MonkeyPatch
) -> None:
    """Test that the version checker can run successfully.

    Args:
        version_dict (Dict[str, str]): The dict containing a pair of linter name and version
        monkeypatch (MonkeyPatch): Monkey patch to patch subprocess.run with.
    """
    version_list = [val for _, val in version_dict.items()]
    monkeypatch.setattr(subprocess, "run", MonkeySubprocessRun(version_list, False))
    version_checker = VersionChecker()

    assert len(version_checker.linter_with_errors) == 0
    assert len(version_checker.missing_errors) == 0
    assert len(version_checker.version_errors) == 0


def test_version_checker_file_not_found(
    capfd: pytest.CaptureFixture, version_dict: Dict[str, str], monkeypatch: MonkeyPatch
) -> None:
    """Test that FileNotFoundErrors are handled correctly

    Args:
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
        version_dict (Dict[str, str]): The dict containing a pair of linter name and version
        monkeypatch (MonkeyPatch): Monkey patch to patch subprocess.run with.
    """
    version_list = [val for _, val in version_dict.items()]
    monkeypatch.setattr(subprocess, "run", MonkeySubprocessRun(version_list, True))
    version_checker = VersionChecker()

    assert len(version_checker.linter_with_errors) == len(version_list)
    assert len(version_checker.missing_errors) == len(version_list)
    assert len(version_checker.version_errors) == 0
    # Test that we print the correct thing
    version_checker.print_errors()
    stdout, _ = capfd.readouterr()
    assert not "not have the correct version" in stdout
    assert "was not found" in stdout
    assert "Install the correct linter" in stdout


def test_version_checker_wrong_version(
    capfd: pytest.CaptureFixture, version_dict: Dict[str, str], monkeypatch: MonkeyPatch
) -> None:
    """Test that FileNotFoundErrors are handled correctly

    Args:
        capfd (pytest.CaptureFixture): The output captured from the file descriptors
        version_dict (Dict[str, str]): The dict containing a pair of linter name and version
        monkeypatch (MonkeyPatch): Monkey patch to patch subprocess.run with.
    """
    # Spoof the first version
    version_dict[list(version_dict.keys())[0]] = "0.0.0"
    version_list = [val for _, val in version_dict.items()]
    monkeypatch.setattr(subprocess, "run", MonkeySubprocessRun(version_list, False))
    version_checker = VersionChecker()

    assert len(version_checker.linter_with_errors) == 1
    assert len(version_checker.missing_errors) == 0
    assert len(version_checker.version_errors) == 1
    # Test that we print the correct thing
    version_checker.print_errors()
    stdout, _ = capfd.readouterr()
    assert "not have the correct version" in stdout
    assert not "was not found" in stdout
    assert "Install the correct linter" in stdout
