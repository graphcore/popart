# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from tempfile import NamedTemporaryFile

import pytest
from lint.config import load_linter_configs


def test_load_linter_config():
    dummy_config = """
{
    "linters": {
        "test-linter": {
            "class": "TestLinter"
        }
    }
}
    """
    with NamedTemporaryFile("w") as f:
        f.write(dummy_config)
        f.seek(0)
        config = load_linter_configs(f.name)

    assert type(config) is dict
    assert "linters" in config
    assert "test-linter" in config["linters"]
    assert config["linters"]["test-linter"]["class"] == "TestLinter"


def test_raises_on_no_config_file():
    filename = "/this/file/doesnt/exist"
    with pytest.raises(AssertionError, match=filename):
        load_linter_configs(filename)


def test_raises_on_missing_linter_field():
    dummy_config = """
{
    "wrong_key": {
        "test-linter": {
            "class": "TestLinter"
        }
    }
}
    """
    with NamedTemporaryFile("w") as f:
        f.write(dummy_config)
        f.seek(0)
        with pytest.raises(KeyError):
            load_linter_configs(f.name)
