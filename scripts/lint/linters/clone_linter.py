# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, List, Sequence
import argparse
import re
from pathlib import Path


def get_op_definition_start(lines: List[str]) -> List[int]:
    r"""Get the line numbers which marks the start of an op definition body.

    Args:
        lines (List[str]): The text to parse

    Returns:
        List[int]: The line numbers where a match has been found
    """
    match_lines = list()
    for nr, line in enumerate(lines):
        pattern = r"^class \w*Op(?!;|\w)"
        match = re.search(pattern, line)
        if match is None:
            continue

        match_lines.append(nr)

    # We now find the starting {
    body_definition_start = list()
    for nr in match_lines:
        for line in lines[nr:]:
            if "{" in line:
                body_definition_start.append(nr)
                break
            nr += 1

    return body_definition_start


def get_closing_bracket(lines: List[str]) -> Optional[int]:
    """Return the line of the closing bracket.

    Args:
        lines (List[str]): List of lines to check

    Returns:
        Optional[int]: The line number the closing bracket was found.
          None if the closing bracket was not found.
    """
    # Get the line number of the closing bracket
    unmatched_brackets = 0
    for nr, line in enumerate(lines):
        unmatched_brackets += line.count("{")
        unmatched_brackets -= line.count("}")
        if unmatched_brackets == 0:
            return nr
    return None


def is_clone_signature_present(lines: List[str]) -> bool:
    """Check whether a clone signature is present.

    Args:
        lines (List[str]): List of lines to check

    Returns:
        bool: True if a clone signature is found, else False
    """
    for line in lines:
        if "unique_ptr<Op> clone()" in line:
            return True
    return False


def check_if_clone_is_defined(filename: str) -> int:
    """Check if unique_ptr<Op> clone() is present for all Op classes in a file.

    Args:
        filename (str): Name of file

    Returns:
        int: 0 if all classes implement Op classes implement clone()
          1 if not
    """
    lines = Path(filename).resolve().read_text("utf-8").split("\n")

    definitions_start = get_op_definition_start(lines)

    ret_val = 0
    for definition_start in definitions_start:
        definition_relative_end = get_closing_bracket(lines[definition_start:])
        if definition_relative_end is None:
            # +1 as line numbers start with 0
            print(
                f"ERROR: Could not find closing bracket for {filename}:{definition_start + 1}"
            )
            ret_val = 1
            continue

        if is_clone_signature_present(lines[definition_start:definition_start +
                                            definition_relative_end]):
            continue

        # +1 as line numbers start with 0
        print(
            f"ERROR: The clone() method needs to be defined in {filename}:{definition_start + 1}"
        )
        ret_val = 1

    return ret_val


def main(argv: Optional[Sequence[str]] = None) -> int:
    r"""Run the clone linter.

    Limitations:
    - The Op definition must be on the form "^class \w*Op(?!;|\w)"

    Args:
        argv (Optional[Sequence[str]]): Parsed arguments

    Returns:
        int: 0 for success, 1 for fail
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    args = parser.parse_args(argv)

    ret_val = 0
    for filename in args.filenames:
        cur_ret_val = check_if_clone_is_defined(filename)
        ret_val |= cur_ret_val

    return ret_val


if __name__ == '__main__':
    raise SystemExit(main())
