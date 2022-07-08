# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from pathlib import Path, PosixPath
from typing import Iterator

import pytest

# See
# https://docs.pytest.org/en/latest/how-to/monkeypatch.html for
# details on monkey patching
from pytest import MonkeyPatch
from scripts.lint.linters.check_python_test_has_cmake_entry import (
    check_for_test_entry,
    is_probably_pytest_file,
    popart_root_dir,
)


class MonkeyPathOpen:
    """Class used to patch the Path.open function."""

    def __init__(self, contents: str) -> None:
        """Set the contents of the Path.

        Args:
            contents (str): Contents of the patch
        """
        self.contents = contents

    def __call__(self) -> "MonkeyFile":
        """Return a MonkeyFile containing self.contents.

        Returns:
            MonkeyFile: Object patching a file
        """
        return MonkeyFile(self.contents)


class MonkeyPathResolve:
    """Class used to patch the Path.resolve function."""

    def __init__(self, path_str: str) -> None:
        """Set the path string.

        Args:
            path_str (str): String returned by resolve
        """
        self.path_str = path_str

    def __call__(self) -> PosixPath:
        """Return a PosixPath.

        Returns:
            PosixPath: The patched PosixPath
        """
        return self.resolve()

    def resolve(self) -> PosixPath:
        """Return a PosixPath of the input.


        Returns:
            PosixPath: The resulting path
        """
        return PosixPath(self.path_str)


class MonkeyFile:
    """Class used to patch a file"""

    def __init__(self, contents) -> None:
        """Set the contents of the patched file.

        Args:
            contents (str): Contents of the patch
        """
        self.contents_list = contents.split("\n")

    def __enter__(self) -> "MonkeyFile":
        """Return itself.

        Returns:
            MonkeyFile: Object returned on enter
        """
        return self

    def __exit__(self, *_) -> None:
        """Return None."""
        return

    def readlines(self) -> Iterator[str]:
        """Return the contents line by line

        Yields:
            Iterator[str]: The iterator of the content lines
        """
        for content in self.contents_list:
            yield content


def test_is_probably_pytest_file(monkeypatch: MonkeyPatch) -> None:
    """Test that is_probably_pytest_file make the correct hits.

    We should expect hits only on:
    - Function names starting with `test`
    - Class names starting with `Test`
    - Any string matching `unittest.main()`

    Args:
        monkeypatch (MonkeyPatch): Monkey patch to patch open
    """
    monkeypatch.setattr(
        Path,
        "open",
        MonkeyPathOpen("def test_my_stuff() -> None:\n" "    return True\n"),
    )
    assert is_probably_pytest_file(Path("Arg not important as it's patched"))

    monkeypatch.setattr(
        Path,
        "open",
        MonkeyPathOpen(
            "class TestMe:\n" "    def a_silly_func():\n" "        return True\n"
        ),
    )
    assert is_probably_pytest_file(Path("Arg not important as it's patched"))

    monkeypatch.setattr(
        Path,
        "open",
        MonkeyPathOpen(
            "def a_silly_func():\n"
            "    return True\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    unittest.main()\n"
        ),
    )
    assert is_probably_pytest_file(Path("Arg not important as it's patched"))

    monkeypatch.setattr(
        Path, "open", MonkeyPathOpen("def a_silly_func():\n" "    return True\n")
    )
    assert not is_probably_pytest_file(Path("Arg not important as it's patched"))


def test_popart_root_dir(monkeypatch: MonkeyPatch) -> None:
    """Test that popart_root_dir is finding the correct popart dir.

    Args:
        monkeypatch (MonkeyPatch): Monkey patch to patch resolve
    """
    monkeypatch.setattr(Path, "resolve", MonkeyPathResolve("/home/user/popart/foo/bar"))
    assert popart_root_dir() == Path("/home/user/popart")

    monkeypatch.setattr(Path, "resolve", MonkeyPathResolve("/home/user/foo/bar"))
    with pytest.raises(RuntimeError) as e_info:
        popart_root_dir()
        assert e_info.value.args[0] == "'popart' not in path tree"


def test_check_for_test_entry(monkeypatch: MonkeyPatch) -> None:
    """Test that check_for_test_entry is returning the correct value.

    We have a valid entry if the following is satisfied:
    - A line in the CMakeLists.txt file has `add_popart_py_unit_test(name_of_file` in it
    - The line may start with one or more `#`
    - There may be whitespaces between  `#` and `add_...`

    Args:
        monkeypatch (MonkeyPatch): Monkey patch to patch open
    """
    lint_path = popart_root_dir().joinpath("foo/bar/baz.py")
    cmakelists_path = popart_root_dir().joinpath("foo/CMakeLists.txt")
    monkeypatch.setattr(
        Path, "open", MonkeyPathOpen(f"#add_popart_py_unit_test({lint_path.stem})\n")
    )
    assert check_for_test_entry(lint_path, cmakelists_path) == 0

    monkeypatch.setattr(
        Path, "open", MonkeyPathOpen(f"add_popart_py_unit_test({lint_path.stem})\n")
    )
    assert check_for_test_entry(lint_path, cmakelists_path) == 0

    monkeypatch.setattr(
        Path, "open", MonkeyPathOpen("add_popart_py_unit_test(something_else.py)\n")
    )
    assert check_for_test_entry(lint_path, cmakelists_path) == 1

    monkeypatch.setattr(
        Path, "open", MonkeyPathOpen(f"add_popart_cpp_unit_test({lint_path.stem})\n")
    )
    assert check_for_test_entry(lint_path, cmakelists_path) == 1
