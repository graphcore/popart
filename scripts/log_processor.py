# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
    A script that can do some very basic log processing.

    The script is currently limited to processing the timestamps in the log. Suppose you have a log like this:

      [2020-07-29 08:59:03.333] First log entry.
      [2020-07-29 08:59:03.338] Second log entry.
      [2020-07-29 08:59:03.458] Third log entry.
      [2020-07-29 08:59:11.670] Fourth log entry.
      [2020-07-29 08:59:11.674] Fifth log entry.

    You can use the '--strip-timestamps' option to remove timestamps altogether. This makes is easier to find functional differences between the logs of two runs:

      [<timestamp>] First log entry.
      [<timestamp>] Second log entry.
      [<timestamp>] Third log entry.
      [<timestamp>] Fourth log entry.
      [<timestamp>] Fifth log entry.

    You can use '--rebase-timestamps' to make all timestamps relative to the first log entry. This is useful for comparing the runtime of two logs:

      [00-00 00:00:00.000] First log entry.
      [00-00 00:00:00.005] Second log entry.
      [00-00 00:00:00.125] Third log entry.
      [00-00 00:00:08.337] Fourth log entry.
      [00-00 00:00:08.341] Fifth log entry.

    Finally, you can use '--delta-timestamps' to make all timestamps relative to the timestamp of the previous log entry. This is useful for finding which steps of the program take up the most time or to compare the performance of a step in two different conditions:

      [<timediff-unavail>] First log entry.
      [00-00 00:00:00.005] Second log entry.
      [00-00 00:00:00.120] Third log entry.
      [00-00 00:00:08.212] Fourth log entry.
      [00-00 00:00:00.004] Fifth log entry.
"""

import argparse
import datetime
import re

datetime_re = re.compile(
    r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{3})\]')


def strip_timestamps(input_file, output_file):
    """
    Change all timestamps to '[<timestamp>]'.

    This is useful if you are comparing two log files with one another but you do not care about timings.
    """
    for line in input_file.readlines():
        line = re.sub(datetime_re, '[<timestamp>]', line)
        output_file.write(line)


def rebase_timestamps(input_file, output_file):
    """
    Change all timestamps to be the time difference from the first log entry.

    This is useful if you are comparing the timings of two log files with one another.
    """
    inception = None
    for line in input_file.readlines():
        match = datetime_re.match(line)
        if match:
            dt = datetime.datetime.strptime(match.groups()[0],
                                            '%Y-%m-%d %H:%M:%S')
            dt = dt.replace(microsecond=int(match.groups()[1]) * 1000)
            if inception is None:
                inception = dt
                line = re.sub(datetime_re, '[00-00 00:00:00.000]', line)
            else:
                td = dt - inception
                td_str = '[{weeks:02d}-{days:02d} {hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}]'.format(
                    weeks=td.days // 7,
                    days=td.days % 7,
                    hours=td.seconds // 3600,
                    minutes=(td.seconds // 60) % 60,
                    seconds=td.seconds % 60,
                    milliseconds=td.microseconds // 1000)
                line = re.sub(datetime_re, td_str, line)
        output_file.write(line)


def delta_timestamps(input_file, output_file):
    """
    Change all timestamps to be the time difference from the previous log entry.

    This is useful if you are comparing the timings of individual steps between two log files.
    """
    prev_dt = None
    for line in input_file.readlines():
        match = datetime_re.match(line)
        if match:
            dt = datetime.datetime.strptime(match.groups()[0],
                                            '%Y-%m-%d %H:%M:%S')
            dt = dt.replace(microsecond=int(match.groups()[1]) * 1000)
            if prev_dt is None:
                line = re.sub(datetime_re, '[<timediff-unavail>]', line)
            else:
                td = dt - prev_dt
                td_str = '[{weeks:02d}-{days:02d} {hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}]'.format(
                    weeks=td.days // 7,
                    days=td.days % 7,
                    hours=td.seconds // 3600,
                    minutes=(td.seconds // 60) % 60,
                    seconds=td.seconds % 60,
                    milliseconds=td.microseconds // 1000)
                line = re.sub(datetime_re, td_str, line)
            prev_dt = dt
        output_file.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input log file")
    parser.add_argument("output_file", help="Output log file")
    parser.add_argument("-s",
                        "--strip-timestamps",
                        action="store_true",
                        help="Strip all timestamps from the log file")
    parser.add_argument(
        "-z",
        "--rebase-timestamps",
        action="store_true",
        help=
        "Change all timestamps in a log file to the time difference relative to the first log entry"
    )
    parser.add_argument(
        "-d",
        "--delta-timestamps",
        action="store_true",
        help=
        "Change all timestamps in a log file to the time difference relative to the first previous log entry"
    )
    args = parser.parse_args()

    with open(args.input_file, 'r') as input_file:
        with open(args.output_file, 'w') as output_file:
            if args.rebase_timestamps:
                rebase_timestamps(input_file, output_file)
            elif args.strip_timestamps:
                strip_timestamps(input_file, output_file)
            elif args.delta_timestamps:
                delta_timestamps(input_file, output_file)
            else:
                print(
                    "No processing option set, defaulting to '--strip-timestamps'"
                )
                strip_timestamps(input_file, output_file)


if __name__ == '__main__':
    main()
