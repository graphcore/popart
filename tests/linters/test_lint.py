# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
import os
from tempfile import TemporaryDirectory

import pytest
from lint.config import LinterConfig
from lint.linters.base_linter import ILinter
from linters.dummy_linters import (DoNothingLinter, FirstLinter, SecondLinter,
                                   ThirdLinter)
from linters.test_base_linter import LinterUnderTest
from run_lint import (LinterOutput, create_linter_configs,
                      create_linters_and_check_for_errors, import_from_string,
                      lint, linter_from_config, run_linters, LinterMessage)


def test_import_from_string():
    join = import_from_string("os.path.join")
    import os
    assert join is os.path.join

    ilinter = import_from_string("lint.linters.ILinter")
    print(type(ilinter))
    assert ilinter is ILinter

    with pytest.raises(ImportError):
        import_from_string("This is not a valid module path")

    with pytest.raises(ImportError):
        import_from_string("this.module.doesnt.exist")

    with pytest.raises(ImportError):
        import_from_string("os.path.foobar")


def test_linter_from_config():
    dummy_config = LinterConfig(name="test_linter",
                                class_="linters.dummy_linters.LinterUnderTest",
                                version="0.0.0")
    linter = linter_from_config(dummy_config)
    assert linter is not None
    assert type(linter) is LinterUnderTest
    assert linter.get_version() == (0, 0, 0)
    assert linter.is_available()

    unavailable_config = LinterConfig(
        name="unavailable_linter",
        class_="linters.dummy_linters.UnavailableLinter",
        version="0.0.0")
    linter = linter_from_config(unavailable_config)
    assert not linter.is_available()


def test_create_linters():
    ok_linters = [
        LinterConfig(
            name="TestLinter",
            class_="linters.dummy_linters.LinterUnderTest",
        ),
        LinterConfig(name="DeleteLinter",
                     class_='linters.dummy_linters.DeleteLinter'),
    ]
    unavailable_linter = LinterConfig(
        name="UnavailableLinter",
        class_="linters.dummy_linters.UnavailableLinter")
    bad_version_linter = LinterConfig(
        name="BadVersionLinter",
        class_="linters.dummy_linters.LinterUnderTest",
        version="2.2.2")
    bad_linters = [unavailable_linter, bad_version_linter]
    configs = ok_linters + bad_linters
    linters, error_msgs = create_linters_and_check_for_errors(configs)
    assert len(linters) == len(ok_linters)
    for linter in linters:
        assert issubclass(type(linter), ILinter)
        assert linter.is_available()

    assert len(error_msgs) == len(bad_linters)
    for msg in error_msgs:
        assert msg
        assert type(msg) is LinterMessage

    linters, error_msgs = create_linters_and_check_for_errors(
        [unavailable_linter])
    assert not linters
    for msg in error_msgs:
        message_text = msg.message
        assert 'Linter UnavailableLinter is not available or installed.' in message_text
        assert 'UnavailableLinterInstallMessage' in message_text

    linters, error_msgs = create_linters_and_check_for_errors(
        [bad_version_linter])
    assert not linters
    for msg in error_msgs:
        message_text = msg.message
        assert 'Version requirement not satisfied for linter BadVersionLinter' in message_text
        assert 'Version required: 2.2.2' in message_text
        assert 'You have: 0.0.0' in message_text


def test_create_configs():
    def write_config(tmpdir, config_str):
        filename = os.path.join(tmpdir, 'linter_config.json')
        with open(filename, 'w') as fp:
            fp.write(config_str)
        return filename

    with TemporaryDirectory() as tmpdir:
        bad_config_str = """
        {
            "linters": {
                "Linter": {
                    "no_class": "in_this_file"
                }
            }
        }
        """
        bad_file = write_config(tmpdir, bad_config_str)
        with pytest.raises(KeyError) as exc:
            create_linter_configs(bad_file)
        assert "Config for linter Linter is missing the required 'class' field." in str(
            exc.value)

        valid_config = {
            "linters": {
                "test-linter": {
                    "class": "linters.dummy_linters.LinterUnderTest",
                    "version": "0.0.0",
                    "exclude": "CMakeLists.txt$",
                    "include": ".py$",
                    "config_file": "some_lint_config.cfg"
                }
            }
        }

        config_file = write_config(tmpdir, json.dumps(valid_config))
        configs = create_linter_configs(config_file)
        assert len(configs) == len(valid_config["linters"])
        for cfg in configs:
            assert type(cfg) is LinterConfig
            # The config file we have created does not exist, but nevertheless
            # we want to test that the directory it would have been loaded from
            # is a valid path on whatever system we're on.
            assert os.path.exists(os.path.dirname(cfg.config_file))


def test_run_linters():
    linters = [FirstLinter(), SecondLinter(), ThirdLinter()]
    noop_linters = [DoNothingLinter()]
    with TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_file')
        with open(filename, 'w'):
            pass  # This makes sure the file is created
        output = run_linters(filename, linters)
        noop_output = run_linters(filename, noop_linters)

    assert type(output) is LinterOutput
    assert output.status_ok
    assert output.original == ''
    assert output.replacement == 'FirstLinter\nSecondLinter\nThirdLinter\n'
    # We currently assume that linters only ever produce one message
    assert len(output.messages) == len(linters)
    for msg in output.messages:
        linters_which_produced_this_message = list(
            filter(lambda x: x.name == msg.linter, linters))
        assert len(linters_which_produced_this_message) == 1
        assert linters_which_produced_this_message[0].get_linter_message(
        ) == msg.message

    assert noop_output.original == noop_output.replacement
    assert len(noop_output.messages) == 0


def test_lint():
    linter_config = {
        "linters": {
            "first-linter": {
                "class": "linters.dummy_linters.FirstLinter"
            },
            "second-linter": {
                "class": "linters.dummy_linters.SecondLinter"
            },
            "third-linter": {
                "class": "linters.dummy_linters.ThirdLinter"
            },
        }
    }
    with TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'linter_config.json')
        file_to_lint = os.path.join(tmpdir, 'file_to_lint')
        with open(config_file, 'w') as fp:
            fp.write(json.dumps(linter_config))
        with open(file_to_lint, 'w'):
            pass

        output = lint(file_to_lint, config_file)
