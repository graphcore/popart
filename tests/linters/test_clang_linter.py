# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import subprocess

import pytest

from lint.config import LinterConfig
from lint.linters import ClangFormatLinter


@pytest.fixture
def linter():
    dummy_config = LinterConfig(
        name="TestLinter",
        class_="scripts.lint.linters.ClangFormatLinter",
    )
    linter = ClangFormatLinter(dummy_config)
    yield linter


def test_is_available(linter):
    assert linter.is_available()


def test_get_version(linter):
    process = subprocess.Popen(["clang-format", "--version"],
                               stdout=subprocess.PIPE,
                               universal_newlines=True)
    stdout, _ = process.communicate()
    process.wait()
    version = linter.get_version()
    version_hash = version[-1]
    assert version_hash in stdout

    version_number = ".".join(str(i) for i in version[:-1])
    assert version_number in stdout


def test_apply_linter(linter):
    # We care less about the actual effects the linter has;
    # we're only checking that the linter actually did _something_.
    dummy_code = """
int main() {
    int i   =   1;
    if (i > 1)
    { int j = 2;}
}
    """

    linted_code = linter.apply('test.cpp', dummy_code)

    assert dummy_code != linted_code
    assert "int i = 1;\n" in linted_code
    assert "if (i > 1) {\n    int j = 2;\n  }\n" in linted_code
