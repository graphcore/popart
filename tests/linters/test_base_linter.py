# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
from lint.config import LinterConfig
from linters.dummy_linters import LinterUnderTest, DeleteLinter


def test_is_applicable():
    config = LinterConfig(
        name="TestLinter",
        class_="TestLinter",
        include=".(cpp|hpp)$",
        exclude=[
            "(.py$)",
            "CMakeLists.txt$",
            "(.*Doxyfile.in$)",
            "(willow/include/popart/vendored/.*)",
        ],
    )
    linter = LinterUnderTest(config)

    included_files = [
        "file.cpp",
        "welldocumentedinterface.hpp",
        "long/path/to/file/header.hpp",
        "/absolute/path/to/file.hpp",
    ]

    excluded_files = [
        "useful_script.py",
        "path/to/robust_and_stable_linter.py",
        "CMakeLists.txt",
        "cmake/is/great/CMakeLists.txt",
        "willow/include/popart/vendored/iseven.hpp",
        "Doxyfile.in",
        "docs/Doxyfile.in",
    ]

    for inc in included_files:
        assert linter.is_applicable(inc)

    for exc in excluded_files:
        assert not linter.is_applicable(exc)


def test_raise_on_invalid_filters():
    config = LinterConfig(name="TestLinter", class_="TestLinter", include=[1, 2, 3])
    with pytest.raises(ValueError):
        _ = LinterUnderTest(config)

    # Not nonetype but invalid regular expression field nonetheless
    config.include = dict()
    with pytest.raises(ValueError):
        _ = LinterUnderTest(config)


def test_gives_correct_arc_message():
    config = LinterConfig(name="DeleteLinter", class_="DeleteLinter")

    linter = DeleteLinter(config)
    linter.apply_lint_function(file_path="", file_contents="BlahBlah")
    assert linter.get_linter_message() == "Wiped the file."
