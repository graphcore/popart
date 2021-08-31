# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Usage:
#     python3 check_test_has_cmake_entry.py path/to/test_file.py
#
# This script checks that pytest test files have a corresponding
# `add_popart_py_unit_test` entry in a neighbouring CMakeLists.txt file.
# If the argument file does not appear to be a pytest file, or there is
# no neighbouring CMakeLists.txt file, the script returns early without error.
# If no cmake entry is found, the script prints a message to stderr
# and returns with return code 1.
from pathlib import Path
import re
import os
import sys


# Check if the file is a pytest test file.
def is_probably_pytest_file(file_path):
    with file_path.open() as f:
        for line in f.readlines():
            if re.match('def test', line):
                return True
            elif re.match('class Test', line):
                return True
            elif re.match('\s*unittest\.main\(\)', line):
                return True
    return False


def popart_root_dir():
    here = Path(__file__).resolve()
    assert 'popart' in here.parts
    while here.parent.name != 'popart':
        here = here.parent
    return here.parent


# Check the CMakeLists.txt file for a call to
# `add_popart_py_unit_test` that takes `lint_path` as a parameter.
# This function exits the progam with return code 1 if no test entry was found.
def check_for_test_entry(lint_path, cmakelists_path):
    with cmakelists_path.open() as f:
        for line in f.readlines():
            # Match 0 or more instances of '# ' to also match commented out entries.
            # We catch commented out entires as these are not tests where the 
            # cmake entry has been forgotten, but usually tests that are broken 
            # and have a task to fix attached.
            pattern = '(#\s*)?'
            # Match instances of 'add_popart_py_unit_test(some_test_name '
            pattern += 'add_popart_py_unit_test\(\S+ '
            # Match only entries with the same lint_path
            pattern += f'{lint_path.name}'
            if re.match(pattern, line):
                return

    test_relative_path = lint_path.relative_to(popart_root_dir())
    cmakelists_relative_path = cmakelists_path.relative_to(popart_root_dir())
    print(
        f"Could not find `add_popart_py_unit_test` entry for '{lint_path.name}' "
        f"in '{cmakelists_relative_path}'. If this is intentional, please exclude the "
        f"path \"({test_relative_path})\" in the popart .arclint file, under the"
        f" section 'popart-test-linter'.",
        file=sys.stderr)
    exit(1)


def main():
    lint_path = Path(sys.argv[1])
    assert lint_path.exists()

    # Only want to lint pytest files.
    if not is_probably_pytest_file(lint_path):
        return

    cmakelists_path = lint_path.parent / 'CMakeLists.txt'
    # If there is no CMakeLists.txt in this directory, just return.
    if not cmakelists_path.exists():
        return

    check_for_test_entry(lint_path, cmakelists_path)


if __name__ == '__main__':
    main()
