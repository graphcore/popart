# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# This test checks through all the header files it can find
# in the argument path and checks the files "#include"
# statements for boost headers.
# Usage:
#   python3 boost_free_interface_test.py path_to_include_folder
import sys
from pathlib import Path


def parse_include(line):
    fname = line.split()[1]
    fname = fname.strip()
    fname = fname[1:-1]
    return fname


def get_includes(path):
    # Encoding is specified as some of the ubuntu buildbots were throwing the error:
    #   UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 1019: ordinal not in range(128)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#include "):
                yield parse_include(line)


def get_headers(include_path):
    headers = []

    # Get all the .h, .hpp, .h.gen, and .hpp.gen files in the include folder.
    # Currently there are only .hpp files, but I've included the others just in case any are added.
    for glob in ("**/*.h", "**/*.hpp", "**/*.h.gen", "**/*.hpp.gen"):
        headers.extend(include_path.glob(glob))

    return headers


def main():
    assert len(sys.argv) == 2
    include_path = Path(sys.argv[1])

    # The include path must exist and must be called include.
    assert include_path.name == "include"
    assert include_path.exists()

    headers = get_headers(include_path)
    # If no headers are found then there has been an error.
    assert len(headers) > 0

    known_failures = ["testdevice.hpp", "mergevarupdates.hpp"]

    # get the names of all the included files
    includes = set()
    for header in headers:
        if header.name in known_failures:
            print(f"Skipping file {header}")
            continue

        for include in get_includes(header):
            print(f"{header}: {include}")
            includes.add(include)

    for include in includes:
        print(f'Checking included file "{include}"')
        assert "boost" not in include


if __name__ == "__main__":
    main()
