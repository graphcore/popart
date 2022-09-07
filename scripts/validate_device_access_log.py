# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
Script to validate 'build/popart/deviceaccess.log'.

This file is generated when running hardware tests with the cmake option
POPART_LOG_DEVICE_ACCESS_IN_TESTS set.

The idea is that this may be able to automatically detect test behaviour
that would be problematic for CI.
"""

import argparse
import datetime
import functools
import io
import sys
from collections import defaultdict, namedtuple
from types import SimpleNamespace


@functools.lru_cache(maxsize=None)
def parse_line(line):
    tokens = [t.strip() for t in line.split(", ")]
    if len(tokens) < 4:
        raise ValueError("expected >=4 items separated by ', '")
    else:
        dt = datetime.datetime.strptime(tokens[0], "%Y-%m-%dT%H:%M:%SZ")
        result = {"timestamp": dt}

        for t in tokens[1:]:
            key, value = t.split(":", 1)
            if key != "ipus":
                try:
                    # Convert to int if possible.
                    result.update({key: int(value)})
                except ValueError:
                    # Otherwise string.
                    result.update({key: value})
            else:
                # Parse set of IPUs.
                ipus = value[1:-1]
                ipus = [int(s) for s in ipus.split(",")] if len(ipus) else []
                result.update({key: ipus})

        if "timestamp" not in result:
            raise ValueError("expected timestamp")
        if "test" not in result:
            raise ValueError("expected 'test'")
        if "event" not in result:
            raise ValueError("expected 'event'")
        if "ipus" not in result:
            raise ValueError("expected 'ipus'")

        return result


def get_parsed_lines_from_file(file):
    for line in file.readlines():
        try:
            yield line, parse_line(line)
        except ValueError as e:
            raise ValueError(f"Unable to parse line: '{line.strip()}' ({e})")


def get_parsed_lines_from_file_name(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        yield from get_parsed_lines_from_file(file)


def get_parsed_lines(input_file):
    if isinstance(input_file, io.StringIO):
        yield from get_parsed_lines_from_file(input_file)
    else:
        yield from get_parsed_lines_from_file_name(input_file)


def get_structured_test_logs(input_file):
    """Convert the textual logs to a structured object.

    As an example, the result would look like:

    {
        "test1": [
            EventData(
                timestamp=datetime.datetime(2022, 1, 4, 15, 26, 42),
                event="attach",
                ipus=[9],
            ),
            EventData(
                timestamp=datetime.datetime(2022, 1, 4, 15, 26, 43),
                event="detach",
                ipus=[9],
            ),
        ],
        "test2": [
            EventData(
                timestamp=datetime.datetime(2022, 1, 4, 15, 26, 42),
                event="select",
                ipus=[1],
            ),
            EventData(
                timestamp=datetime.datetime(2022, 1, 4, 15, 26, 44),
                event="try-attach-start",
                ipus=[1],
            ),
            EventData(
                timestamp=datetime.datetime(2022, 1, 4, 15, 26, 45),
                event="try-attach-success",
                ipus=[1],
            ),
        ],
    }
    """
    test_logs = defaultdict(list)
    EventData = namedtuple("EventData", ["timestamp", "event", "ipus"])

    for _, parsed_line in get_parsed_lines(input_file):
        test = parsed_line["test"]
        timestamp = parsed_line["timestamp"]
        event = parsed_line["event"]
        ipus = parsed_line["ipus"]

        event_data = EventData(timestamp, event, ipus)
        test_logs[test].append(event_data)

    return test_logs


def validate_events(args):
    """
    Check each test follows the NFA described below.

            (init)
              |
    .-------->|  <... expected final state
    |         |
    |         |
    |         |'---------------------------.
    |         |                            v
    |         |                          select
    |         |                            |
    |         |.--------------------------'|
    |         v                            v
    |       attach                  try-attach-start
    |         |                            |
    |         |                            |'------------------.
    |         |                            v                   v
    |         |                    try-attach-success   try-attach-fail
    |         |                            |                   |
    |         |.---------------------------'                   X (error)
    |         |
    |         |'------------.
    |         |             |
    |         v             v
    |       detach   detach-on-destruct
    |         |             |
    '---------'<------------'

    If tests don't follow this order either a test is not detaching (maybe it
    crashed, or maybe it attaches to a new device before detaching from a
    previous device. Either way, it's bad because it can lead to test failures.
    """

    state_map = {}
    status = 0

    for line, parsed_line in get_parsed_lines(args.input_file):
        test = parsed_line["test"]
        this_event = parsed_line["event"]

        def fail(error_msg):
            print(f"error: {error_msg}:\n - {line.strip()}")
            nonlocal status
            status += 1

        def ensure_event(allowed_events, last_event=None):
            if this_event not in allowed_events:
                allstr = "[" + ", ".join([f"'{e}'" for e in allowed_events]) + "]"
                if last_event is None:
                    fail(f"expected one of {allstr} as test's first event")
                else:
                    fail(f"expected one of {allstr} to follow '{last_event}'")

        if test not in state_map.keys():
            ensure_event(["attach", "select"])
        else:
            last_event = state_map[test]["event"]
            if last_event == "attach":
                ensure_event(["detach", "detach-on-destruct"], last_event)
            elif last_event == "select":
                ensure_event(["attach", "try-attach-start"], last_event)
            elif last_event == "try-attach-start":
                ensure_event(["try-attach-success", "try-attach-fail"], last_event)
            elif last_event == "try-attach-success":
                ensure_event(["detach", "detach-on-destruct"], last_event)
            elif last_event == "try-attach-fail":
                fail("encountered 'try-attach-fail")
            elif last_event == "detach":
                ensure_event(["attach", "select"], last_event)
            elif last_event == "detach-on-destruct":
                ensure_event(["attach", "select"], last_event)

        state_map[test] = parsed_line
        # if state_map

    return status


def validate_explicit_detach(args):
    """Validate that each test handles device access properly.

    This means that each test explicitly detaches from the devices it is attached to,
    and that no test is attached to multiple devices at the same time.
    """
    test_logs = get_structured_test_logs(args.input_file)
    all_error_messages = []
    num_failed = 0

    def multi_attach_error_message(attached_ipus):
        return (
            f"  - Test attached to IPUs {attached_ipus[-1]}, while also being attached "
            f"to {attached_ipus[:-1]}."
        )

    def no_dettach_error_message(attached_ipus):
        return (
            "  - Test finished without explicitly detaching from the following IPUs - "
            f"{attached_ipus}."
        )

    for test, event_datas in test_logs.items():
        attached_ipus = []
        implicitly_detached_ipus = []
        test_error_messages = []

        for _, event, ipus in event_datas:
            if event == "attach":
                attached_ipus.append(ipus)
                if len(attached_ipus) > 1:
                    test_error_messages.append(
                        multi_attach_error_message(attached_ipus)
                    )
            elif event == "detach":
                if ipus in attached_ipus:
                    attached_ipus.remove(ipus)
            elif event == "detach-on-destruct":
                if ipus in attached_ipus:
                    attached_ipus.remove(ipus)
                    implicitly_detached_ipus.append(ipus)

        if len(attached_ipus) or len(implicitly_detached_ipus):
            test_error_messages.append(
                no_dettach_error_message(attached_ipus + implicitly_detached_ipus)
            )

        if len(test_error_messages):
            num_failed += 1
            all_error_messages.append(
                f"Test '{test}' did not handle device access properly:"
            )
            all_error_messages.extend(test_error_messages)
            if args.verbose:
                all_error_messages.append(
                    "  Here's the device access log for this test:"
                )
                for timestamp, event, ipus in event_datas:
                    all_error_messages.append(
                        f"    - {timestamp.isoformat()}: event={event}, ipus={ipus}"
                    )

    if len(all_error_messages):
        print(
            "Error: Some tests did not handle device access properly. Please ensure "
            "that all tests explicitly detach from the devices they use, and that no "
            "test attaches to more than one device at a time. Here's more information:"
        )
        print(*all_error_messages, sep="\n")

    num_tests = len(test_logs)
    success_rate = int(100 * (num_tests - num_failed) / num_tests)
    print(
        f"\n{success_rate}% tests handled device access properly, {num_failed} tests "
        f"out of {num_tests} did not"
    )

    return num_failed


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        title="subcommands", description="valid subcommands", help="additional help"
    )

    subparser = subparsers.add_parser("validate_explicit_detach")
    subparser.set_defaults(func=validate_explicit_detach)
    subparser.add_argument("--verbose", action="store_true")

    subparser = subparsers.add_parser("validate_events")
    subparser.set_defaults(func=validate_events)

    parser.add_argument("input_file", help="Input log file")

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


def test_validate_explicit_detach_valid():
    file = io.StringIO(
        """0001-01-01T00:00:00Z, test:test1, event:attach, ipus:[9]
        0001-01-01T00:00:00Z, test:test1, event:detach, ipus:[9]"""
    )
    assert validate_explicit_detach(SimpleNamespace(input_file=file, verbose=True)) == 0


def test_validate_explicit_detach_invalid_0():
    file = io.StringIO(
        """0001-01-01T00:00:00Z, test:test1, event:attach, ipus:[9]
        0001-01-01T00:00:00Z, test:test1, event:attach, ipus:[8]
        0001-01-01T00:00:00Z, test:test1, event:detach, ipus:[8]
        0001-01-01T00:00:00Z, test:test1, event:detach, ipus:[9]"""
    )
    assert validate_explicit_detach(SimpleNamespace(input_file=file, verbose=True)) == 1


def test_validate_explicit_detach_invalid_1():
    file = io.StringIO(
        """0001-01-01T00:00:00Z, test:test1, event:attach, ipus:[9]
        0001-01-01T00:00:00Z, test:test2, event:attach, ipus:[8]
        0001-01-01T00:00:00Z, test:test2, event:detach-on-destruct, ipus:[8]
        0001-01-01T00:00:00Z, test:test1, event:detach, ipus:[9]"""
    )
    assert validate_explicit_detach(SimpleNamespace(input_file=file, verbose=True)) == 1


def test_validate_explicit_detach_invalid_2():
    file = io.StringIO(
        """0001-01-01T00:00:00Z, test:test1, event:attach, ipus:[9]
        0001-01-01T00:00:00Z, test:test2, event:attach, ipus:[8]
        0001-01-01T00:00:00Z, test:test3, event:attach, ipus:[7]
        0001-01-01T00:00:00Z, test:test3, event:attach, ipus:[6]
        0001-01-01T00:00:00Z, test:test3, event:detach, ipus:[7]
        0001-01-01T00:00:00Z, test:test3, event:detach, ipus:[6]
        0001-01-01T00:00:00Z, test:test1, event:detach-on-destruct, ipus:[9]"""
    )
    assert validate_explicit_detach(SimpleNamespace(input_file=file, verbose=True)) == 3


def test_valid_log():
    file = io.StringIO(
        """2022-01-04T15:26:42Z, test:test1, event:attach, ipus:[9]
        2022-01-04T15:26:42Z, test:test2, event:select, ipus:[1]
        2022-01-04T15:26:43Z, test:test1, event:detach, ipus:[9]
        2022-01-04T15:26:44Z, test:test2, event:try-attach-start, ipus:[1]
        2022-01-04T15:26:45Z, test:test2, event:try-attach-success, ipus:[1]"""
    )
    assert validate_events(SimpleNamespace(input_file=file)) == 0


def test_invalid_log():
    file = io.StringIO(
        """2022-01-04T15:26:42Z, test:test1, event:attach, ipus:[9]
        2022-01-04T15:26:42Z, test:test1, event:attach, ipus:[5]
        2022-01-04T15:26:43Z, test:test1, event:detach, ipus:[9]
        2022-01-04T15:26:43Z, test:test1, event:detach, ipus:[5]"""
    )
    assert validate_events(SimpleNamespace(input_file=file)) == 2
