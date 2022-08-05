# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest


def pytest_addoption(parser: "_pytest.config.argparsing.Parser") -> None:  # type: ignore
    """Pytest config to add an option to command line args.

    See https://docs.pytest.org/en/7.1.x/example/simple.html.

    Args:
        parser (_pytest.config.argparsing.Parser): A pytest parser that parses
            the command line options.
    """
    parser.addoption("--device-type", action="store", default="Cpu")


@pytest.fixture(autouse=True, scope="function")
def set_test_device(
    request: "_pytest.fixtures.SubRequest",  # type: ignore
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pytest fixture to set the `TEST_TARGET` env variable for use in the
    test_util.create_test_device function.

    See https://docs.pytest.org/en/6.2.x/fixture.html.

    This fixture is run on every pytest test in this directory and subdirectories
    via the autouse=True argument. Using the monkeypatch object will also stop
    environment variables from leaking.
    See https://docs.pytest.org/en/6.2.x/monkeypatch.html#monkeypatching-environment-variables

    Args:
        request (_pytest.fixtures.SubRequest): A pytest SubRequest object used
            to get the command line options.
        monkeypatch (pytest.MonkeyPatch): A pytest monkeypatch object.
    """
    device_type = request.config.getoption("--device-type")
    monkeypatch.setenv("TEST_TARGET", str(device_type))
