# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
from lint.config import LinterConfig
from lint.linters import YapfLinter


@pytest.fixture
def linter():
    dummy_config = LinterConfig(name="TestLinter",
                                class_="lint.linters.YapfLinter",
                                config_file=".style.yapf",
                                version='1.2.3')
    linter = YapfLinter(dummy_config)
    yield linter


def test_is_available(linter):
    assert linter.is_available()


def test_install_instructions(linter):
    assert linter.install_instructions('1.2.3') == 'pip install yapf==1.2.3'


def test_apply_linter(linter: YapfLinter):
    code = ("if __name__ == '__main__':\n"
            "   x : int     = 5\n"
            "   print(    'Hello World'    )\n")
    linted_code = linter.apply('file.py', code)

    assert code != linted_code


def test_get_version(linter):
    version = linter.get_version()
    assert type(version) is tuple
    assert all(type(i) is int for i in version)
    try:
        import yapf
        actual_version = yapf.__version__
    except ImportError:
        pytest.skip()

    assert ".".join(str(i) for i in version) == actual_version
