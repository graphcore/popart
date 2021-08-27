# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import List

from lint.config import LinterConfig
from lint.linters import ILinter

__all__ = ["CopyrightLinter"]

GC_COPYRIGHT_NOTICE_PATTERN = r"[\/*# ]+Copyright (\xa9|\(c\)) (?P<year>\d{4}) Graphcore Ltd\. All rights reserved\."
GC_COPYRIGHT_NOTICE = f"Copyright (c) {datetime.now().year} Graphcore Ltd. All rights reserved.\n"

LINES_CHECKED = 5

COMMENT_DELIMITERS = {
    r"\.(py|sh|cmake)$": "#",
    r"\.(c|cpp|C|cc|c\+\+|cxx|h|hpp|php)$": "//"
}


class CopyrightLinter(ILinter):
    """"This linter inserts a Graphcore copyright notice into any source file 
    which does not already contain one, determining the format of the notice
    from the source file name.
    """

    def __init__(self, config: LinterConfig):
        super().__init__(config)

    def apply_lint_function(self, file_path: str, file_contents: str):
        """Insert a copyright notice into file_contents if not already present.

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
        assert file_path is not None, "Copyright linter requires the path to the file."

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
            new_contents = GC_COPYRIGHT_NOTICE
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
        for regex, delim in COMMENT_DELIMITERS.items():
            if re.search(regex, filename):
                break
        return delim + " " + GC_COPYRIGHT_NOTICE

    def _partial_match(self, lines: List[str]):
        """Check the lines of the file for a comment which is a close match to the 
        copyright notice, but is not exact enough to be matched by a regular
        expression.
        
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

    def get_version(self):
        # This linter doesn't really have a version.
        # Increment below if your heart so desires.
        return (0, 0, 0)

    def is_available(self):
        return True

    def install_instructions(self) -> str:
        return "No install required."
