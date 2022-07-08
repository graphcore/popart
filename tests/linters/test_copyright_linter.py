# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from datetime import datetime

import pytest

from scripts.lint.linters.copyright_linter import CopyrightLinter

expected_copyright_notice = (
    f"Copyright (c) {datetime.now().year} Graphcore Ltd. All rights reserved.\n"
)


@pytest.fixture
def linter():
    linter = CopyrightLinter()
    yield linter


def test_linter_does_nothing_for_valid_notice(linter: CopyrightLinter):
    code = "# " + expected_copyright_notice
    code = code + "\n\n\n\n\n\n\n\n\n\nThis should still be here"
    linted_code = linter._determine_linter_message("scripts/good_file.py", code)

    assert code == linted_code


def test_linter_inserts_notice(linter: CopyrightLinter):
    test_code = [
        "int myFunction(int a, int b);",
        "\n\nprint('hello')",
        "echo $PATH\n\n\n\nps -aux",
    ]
    test_file_extensions = [".hpp", ".py", ".sh"]
    comment_delims = ["//", "#", "#"]

    for code, ext, delim in zip(test_code, test_file_extensions, comment_delims):
        linted_code = linter._determine_linter_message("file" + ext, code)
        assert delim + " " + expected_copyright_notice + code in linted_code


def test_linter_replaces_incorrect_year(linter: CopyrightLinter):
    outdated_notice = "# " + expected_copyright_notice.replace(
        str(datetime.now().year), "2000"
    )
    correct_notice = "# " + expected_copyright_notice

    print_stmt = "print('hello')"
    code = outdated_notice + print_stmt
    linted_code = linter._determine_linter_message("test.py", code)

    assert correct_notice in linted_code
    assert outdated_notice not in linted_code
    assert print_stmt in linted_code

    number_of_lines = 5
    for i in range(number_of_lines):
        file_header = ["#\n"] * number_of_lines
        file_header[i] = outdated_notice
        code = "".join(file_header) + print_stmt
        linted_code = linter._determine_linter_message("test.py", code)

        assert correct_notice in linted_code
        assert outdated_notice not in linted_code
        assert code.index(outdated_notice) == linted_code.index(correct_notice)


def test_linter_replaces_partial_match(linter: CopyrightLinter):
    # These notices don't match the copyright regular expression, but
    # are reasonably valid notices regardless. We expect them to be
    # replaced with a notice that matches the consistency of other
    # notices, instead of adding a new notice at the top of the file
    year = datetime.now().year
    notices = [
        f"// Copyright(c) {year} Graphcore ltd.all rights reserved.",
        f"// Copyright   (c)    {year}    Graphcore ltd.   All rights reserved.",
        f"// copright (c) {year} graphcore ltd. all rights reserved.",
        f"// copright {year} graphcore ltd all rights reserved",
        f"// Copyright {year} Graphcore. All rights reserved.",
        "// Copright (c) Graphcore. All rights resreved.",
        f"// Copyright (c) {year} Graphcore Ltd\n",
        f"// Copright (c) {year} Graphcore",
        "// copyright graphcore all rights reserved",
        "// coprihgt grapcroe lts. all rihts resrved",
        f"// Copyright Graphcore {year}. All rights reserved.",
        f"// Copyright Graphcore {year} (c). All rights reserved.",
        f"// Copyright {year} (c) Graphcore. All rights reserved.",
    ]

    for wrong_notice in notices:
        linted_code = linter._determine_linter_message("test_file", wrong_notice)
        assert expected_copyright_notice in linted_code
        assert wrong_notice not in linted_code
        assert len(linted_code.split("\n")) == 2


def test_linter_works_for_empty_files(linter: CopyrightLinter):
    linted_code = linter._determine_linter_message("test.sh", "")

    assert expected_copyright_notice in linted_code


def test_inserts_notice_after_shebang(linter: CopyrightLinter):
    shebang = "#!/bin/bash\n"
    code = shebang + "echo 'Hello'"
    linted_code = linter._determine_linter_message("test.sh", code)

    assert expected_copyright_notice in linted_code
    assert linted_code.index(shebang) < linted_code.index(expected_copyright_notice)
