#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
import argparse
from datetime import datetime
from difflib import SequenceMatcher
from typing import List
from typing import Optional
from typing import Sequence
from pathlib import Path

__all__ = ["CopyrightLinter"]

GC_COPYRIGHT_NOTICE_PATTERN = r"[\/*# ]+Copyright (\xa9|\(c\)) (?P<year>\d{4}) Graphcore Ltd\. All rights reserved\."
GC_COPYRIGHT_NOTICE = f"Copyright (c) {datetime.now().year} Graphcore Ltd. All rights reserved.\n"

LINES_CHECKED = 5

COMMENT_DELIMITERS = {
    r"\.(capnp|cmake|py|sh|txt|yaml|yml)$": "#",
    r"\.(c|C|cc|cpp|cxx|c\+\+|h|hpp|php)$": "//"
}


class CopyrightLinter:
    """"Linter which inserts a Graphcore copyright if non-existent.

    The format of the notice is determined from the source file extension.

    The file contents are processed in one of three ways:
    - The file contains a valid notice already, so it is not modified.
    - The file contains no string matching a notice exactly or partially,
      so a new notice is inserted at the top of the file (unless the first
      line is a shebang, in which case we insert one immediately after the
      first line)
    - The file contains a string which matches a copyright notice partially,
      but not exactly. This might happen is the notice contains a typo, or
      some syntactical differences for example. In order to avoid two notices
      which are technically different, but appear almost idenitcal to the eye
      we replace the partially-matching notice with a correct one.
    In a more rare case, a notice might match exactly, but have in incorrect or
    outdated year in the notice. Here, we also replace the notice with an
    updated year.
    """

    def __init__(self):
        self._linter_message = None

    def apply_lint_function(self, file_path: str) -> int:
        """Lint function to be called by pre-commit.

        Args:
            file_path (str): The path to the file to be linted.

        Returns:
            int: If there is no modification to the source file the function returns 0,
              else it will rewrite the file and return 1
        """
        path = Path(file_path)
        file_contents = Path(file_path).read_text(encoding="utf-8")
        new_contents = self._determine_linter_message(file_path, file_contents)

        if self._linter_message is None:
            return 0
        else:
            print(f"Fixing ERROR in {file_path}: {self._linter_message}")
            path.write_text(new_contents, encoding="utf-8")
            return 1

    def set_linter_message(self, message: str) -> None:
        """"Set the message describing and/or explaining the changes applied by this linter."""
        self._linter_message = message

    def _determine_linter_message(self, file_path: str,
                                  file_contents: str) -> str:
        """Determine the linter message (if any) and return the (possibly modified) file content.

        Args:
            file_path (str): The path to the file to be linted.
            file_contents (str): The content of the file.

        Returns:
            str: The file contents which has been modified in the case of a missing copyright notice
        """
        lines = file_contents.splitlines(keepends=True)
        # We only search for the copyright notice in the first
        # n lines in a file
        new_contents = file_contents
        target_lines = lines[:LINES_CHECKED]
        match, index = self._match_copyright(target_lines)
        partial_index = self._partial_match(target_lines)
        if match:
            year = int(match.group('year'))
            # The copyright notice must be after graphcore
            # was founded and can't be in the future
            current_year = datetime.now().year
            if year < 2016 or year > current_year:
                self.set_linter_message(
                    f"Invalid year in copyright notice. Should be <={current_year} and >=2016."
                )
                new_contents = self._insert_copyright_notice(
                    file_path, lines, index=index, replace_notice=True)
        elif partial_index != -1:
            self.set_linter_message(
                f"Copyright notice should be: '{GC_COPYRIGHT_NOTICE.strip()}'")
            new_contents = self._insert_copyright_notice(file_path,
                                                         lines,
                                                         index=partial_index,
                                                         replace_notice=True)

        elif lines:
            self.set_linter_message("No copyright notice in file.")
            new_contents = self._insert_copyright_notice(file_path, lines)
        # If its an empty file then we just include the notice
        else:
            self.set_linter_message("No copyright notice in file.")
            new_contents = self._determine_notice_from_name(file_path)
        return new_contents

    def _head(self, file_contents: str) -> List[str]:
        return file_contents.splitlines(keepends=True)

    def _match_copyright(self, lines: List[str]):
        for i, line in enumerate(lines):
            m = re.search(GC_COPYRIGHT_NOTICE_PATTERN, line)
            if m:
                return m, i
        return False, -1

    def _insert_copyright_notice(self,
                                 file_path: str,
                                 lines: List[str],
                                 index: int = 0,
                                 replace_notice=False):
        notice = self._determine_notice_from_name(file_path)
        if replace_notice:
            lines[index] = notice
        # If the first line is a shebang
        # we insert the notice just after it
        elif ("#!" in lines[0] or "<?php" in lines[0]) and index == 0:
            lines.insert(1, notice)
        else:
            lines.insert(index, notice)
        return "".join(lines)

    def _determine_notice_from_name(self, full_path: str) -> str:
        """Determine how the opyright notice comment should appear in a given file.
        This depends on the filename extension of the file because we use this to
        find out what the comment delimiter should be. For example, if a filename
        ends in .py we know the commend delimiter should be '#'.

        If we cannot find a filename extension pattern in COMMENT_DELIMETERS which
        matches the filename, fall back on '//'. Ideally the exclude and include
        filters should guarantee that the file we are linting has an extension which
        we support.
        """
        split = full_path.rsplit("/", 1)
        if len(split) == 1:
            filename = split[0]
        else:
            _, filename = split
        filename.replace(".in", "")

        delim = ""
        for regex, delim in COMMENT_DELIMITERS.items():
            if re.search(regex, filename):
                break

        if delim == "":
            raise RuntimeError(
                f"Could not find comment delimiter for {full_path}.\n"
                "Possible solution: Add the file type delimiter to the "
                "'COMMENT_DELIMITERS' variable.\n")

        return delim + " " + GC_COPYRIGHT_NOTICE

    def _partial_match(self, lines: List[str]):
        """
        Check the lines of the file for a comment which is a close match to the copyright notice.

        This is often useful for files which do contain copyright notices, but they
        have some syntactical or format errors which cause them not to match the
        notice regular expression. Instead of  inserting a new notice, and creating
        a file which contains two notices that appear to be basically the same we
        instead replace any line in the file which reasonably looks like a good
        copyright notice to the eye, avoiding writing two notices.
        """
        for i, line in enumerate(lines):
            s = SequenceMatcher(None,
                                GC_COPYRIGHT_NOTICE.upper().strip(),
                                line.upper().strip())
            # Accroding to the python documentation, a ratio() value
            # over 0.6 means the sequences are close matches.
            # https://docs.python.org/3/library/difflib.html#sequencematcher-examples
            if s.ratio() > 0.6:
                return i
        return -1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    args = parser.parse_args(argv)

    ret_val = 0
    for filename in args.filenames:
        copyright_linter = CopyrightLinter()
        cur_ret_val = copyright_linter.apply_lint_function(filename)
        ret_val |= cur_ret_val

    return ret_val


if __name__ == '__main__':
    raise SystemExit(main())
