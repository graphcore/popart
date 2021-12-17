# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
    A script to validate the 'build/popart/deviceaccess.log' file generated when
    running hardware tests with the cmake option
    POPART_LOG_DEVICE_ACCESS_IN_TESTS set.

    The idea is that this may be able to automatically detect test behaviour
    that would be problematic for CI.
"""

import argparse
import datetime
import io
import sys


def parse_line(line):
    tokens = [t.strip() for t in line.split(", ")]
    if (len(tokens) < 4):
        raise ValueError("expected >=4 items separated by ', '")
    else:
        dt = datetime.datetime.strptime(tokens[0], '%Y-%m-%dT%H:%M:%SZ')
        result = {'timestamp': dt}

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
                ipus = [int(s) for s in value[1:-1].split(",")]
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


def validate_events(input_file):
    """
    Check each test follows the following NFA:

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
    validLog = True

    for line in input_file.readlines():
        try:
            parsed_line = parse_line(line)

            test = parsed_line['test']
            this_event = parsed_line['event']

            def fail(error_msg):
                print(f"error: {error_msg}:\n - {line.strip()}")
                nonlocal validLog
                validLog = False

            def ensure_event(allowed_events, last_event=None):
                if this_event not in allowed_events:
                    allstr = "[" + ", ".join(
                        [f"'{e}'" for e in allowed_events]) + "]"
                    if last_event is None:
                        fail(f"expected one of {allstr} as test's first event")
                    else:
                        fail(
                            f"expected one of {allstr} to follow '{last_event}'"
                        )

            if test not in state_map.keys():
                ensure_event(['attach', 'select'])
            else:
                last_event = state_map[test]['event']
                if last_event == 'attach':
                    ensure_event(['detach', 'detach-on-destruct'], last_event)
                elif last_event == 'select':
                    ensure_event(['attach', 'try-attach-start'], last_event)
                elif last_event == 'try-attach-start':
                    ensure_event(['try-attach-success', 'try-attach-fail'],
                                 last_event)
                elif last_event == 'try-attach-success':
                    ensure_event(['detach', 'detach-on-destruct'], last_event)
                elif last_event == 'try-attach-fail':
                    fail(f"encountered 'try-attach-fail")
                elif last_event == 'detach':
                    ensure_event(['attach', 'select'], last_event)
                elif last_event == 'detach-on-destruct':
                    ensure_event(['attach', 'select'], last_event)

            state_map[test] = parsed_line
            #if state_map
        except ValueError as e:
            raise ValueError(f"Unable to parse line: '{line.strip()}' ({e})")

    return validLog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input log file")
    args = parser.parse_args()

    with open(args.input_file, 'r') as input_file:
        return validate_events(input_file)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)


def test_valid_log():
    file = io.StringIO(
        """2022-01-04T15:26:42Z, test:test1, event:attach, ipus:[9]
        2022-01-04T15:26:42Z, test:test2, event:select, ipus:[1]
        2022-01-04T15:26:43Z, test:test1, event:detach, ipus:[9]
        2022-01-04T15:26:44Z, test:test2, event:try-attach-start, ipus:[1]
        2022-01-04T15:26:45Z, test:test2, event:try-attach-success, ipus:[1]"""
    )
    assert validate_events(file)


def test_invalid_log():
    file = io.StringIO(
        """2022-01-04T15:26:42Z, test:test1, event:attach, ipus:[9]
        2022-01-04T15:26:42Z, test:test1, event:attach, ipus:[5]
        2022-01-04T15:26:43Z, test:test1, event:detach, ipus:[9]
        2022-01-04T15:26:43Z, test:test1, event:detach, ipus:[5]""")
    assert not validate_events(file)
