# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
    TL;DR: 
    1) Include POPART_ENABLE_COVERAGE=ON in POPART_CMAKE_ARGS when building PopART,
    2) Install gcovr locally using `pip install gcovr`
    3) Check out the command-line options below or run `python view_coverage.py -h`

    ----
    This script combines under one CLI the ability to run tests and view coverage of
    the PopART unit tests.

    In order to instrument the PopART source code under willow/ for code coverage, 
    you must enable the POPART_ENABLE_COVERAGE flag when running CMake. This enables
    specific compiler flags during compilation, resulting in coverage output being
    produced when the source is compiled and executed.
        Compiling the code with these flags enabled will produce .gcno files in the
    same directory as the object files. These .gcno files contain one half of the
    coverage data. The other half of the data comes from .gcda files that are generated
    when you run the instrumented code, with a separate .gcda file for each object file. 
    Each time you run the program, the execution counts are summed into any existing
    .gcda files, so be sure to clean files from previous runs using the --clean flag if 
    you do not want their contents to be included.

    Both .gcno and .gcda files are necessary to produce a coverage report using the 
    command-line tool gcovr. If you haven't already, install it by running:
        pip install gcovr

    This command line tool parses the .gcno and .gcda files to produce a coverage report
    in a variety of formats, such as JSON, HTML, CSV or plaintext.

    https://gcovr.com/en/stable/index.html

    --- COMMAND OPTIONS

    This command-line options of this script are as follows:
     * -p/--format {all, concise, html, csv, json} specifies the output format for a
        coverage report. This is the output format for any invocation of the script
        i.e for gcovr or for test. If you want any other output, such as JSON or XML
        then use the  `gcovr --advanced=` option. 
     * -c/--clean scans the willow CMakeFiles and deletes .gcda files, which result
        from execution of code instrumented with coverage compiler flags. It is a good
        idea to run this before running a different suite of tests, for example if 
        running unit tests after having executed the code using the full test suite
        on a previous run.
     *  -d/--directory specifies the directory where PopART will be built. This script
        will try to infer it from the CBT_BUILDTREE variable, however you may specify 
        it yourself if you don't want to activate the build tree.
     * -o/--output specifies the directory where the `.coverage` folder containing any 
        report files is placed. This defaults to the popart/ directory if omitted.

        The following arguments are applicable to the `test` subcommand:

        *  -r/--ctest-regexp <regex> run any tests matching the regex pattern. 
            This would be the same type of pattern as you would provide when 
            running `ctest -R <regex>`
        *  -f/--filter <gcovr-regexp> displays coverage output for
            source files which match <gcovr-regexp>. 
        
        The `test` subcommand is useful if you are working on testing a specific file 
        and only want to run and view test coverage for that file. For example, if you
        are looking to measure unit test coverage for topocons.cpp, you might run: 
            python view_coverage.py test -r unittest -f .*topocons.cpp

        The following arguments are applicable to the `gcovr` subcommand:

        *  -f/--filter <gcovr-regexp> shows line coverage for all files matching
            gcovr-regexp, similarly to test -f.

        * -e/--advanced allows you to pass a custom string of command line args for
            gcovr, as documented here:
                https://gcovr.com/en/stable/guide.html#the-gcovr-command.
            The flags MUST be enclosed in quotes otherwise argparse will treat them as
            literal flags for this script!! For example, if we want to produce a coverage
            report in XML:
                python view_coverage.py gcovr --advanced="--xml coverage.xml"
"""

import json
import os
import re
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path, PosixPath
from typing import List, Optional

output_types = {
    "all": "--txt",
    "concise": "--json-summary",
    "json": "--json-summary",
    "html":
    "--html-self-contained --html-details {output_dir}/.coverage/coverage.html",
    "csv": "--csv -o {output_dir}/.coverage/coverage.csv",
    "cobertura": "--xml -o {output_dir}/.coverage/coverage.xml"
}


class GCovrRunnerParser:
    def __init__(self, output: str, workspace_dir: PosixPath,
                 build_dir: PosixPath) -> None:
        if output not in output_types:
            raise AssertionError(
                f"Invalid output type {output}. Must be one of {output_types}")
        self.output = output
        self.workspace_dir = workspace_dir
        self.build_dir = build_dir
        self._excluded_paths = []
        self._filter = []
        self._output_path = workspace_dir

    def set_output_path(self, path: PosixPath) -> None:
        if not (path.exists() and path.is_dir()):
            raise RuntimeError(
                f"Output path {path} does not exist or is not a directory.")
        self._output_path = path

    def create_report(self, args=[]) -> str:
        coverage = self._run_gcovr(args)
        return self._parse_gcovr_output(coverage)

    def _create_coverage_folder(self):
        coverage_dir = self._output_path.joinpath(".coverage")
        if not os.path.exists(coverage_dir):
            os.mkdir(coverage_dir)

    def _run_gcovr(self, args=[]) -> str:
        gcovr_path = shutil.which("gcovr")
        if gcovr_path is None:
            raise RuntimeError(
                "Command 'gcovr' not found. Please install it using pip install gcovr."
            )

        flag = output_types[self.output]
        if self.output in {"html", "csv", "cobertura"}:
            self._create_coverage_folder()
            # We keep the output flags as a string so that we can use .format
            # to specify any output path
            flag = flag.format(output_dir=self._output_path)

        src_files_dir = self.workspace_dir.joinpath("willow/src")
        command = [
            'gcovr', '-r', src_files_dir,
            build_files_dir(self.build_dir), '-j16'
        ]
        # The only time we have args is when the user
        # has supplied the --advanced option
        command += args if args else flag.split()
        if self._filter:
            command += self._filter
        # Exclude system library code and headers just in case
        command += [
            '--exclude-directories', '/usr/include/.*', *self._excluded_paths
        ]
        return bash(command, log=False, cwd=self.workspace_dir)

    def _convert_output_to_flag(self) -> str:
        flag = output_types[self.output]
        flag = flag.format(output_dir=self._output_path)
        return flag

    def _parse_gcovr_output(self, coverage: str) -> None:
        if self.output == "concise":
            cov = json.loads(coverage)
            report_str = (f"Line coverage: {cov['line_percent']}%\n"
                          f"Branch coverage: {cov['branch_percent']}%")
            return report_str
        return coverage

    def set_filter(self, filter_regex: str) -> None:
        if filter_regex:
            filter_regex = filter_regex.strip()
            excludes = self._get_list_of_excluded_directories(filter_regex)
            self._excluded_paths = excludes
            self._filter = ["-f", filter_regex]

    def _get_list_of_excluded_directories(self, file_regex: str) -> List[str]:
        """
        Produce a list of flags which forces gcovr to only search directories
        which contain files matching file_regex for raw gcov files.

        Having gcovr search all the CMakeBuild files when a filter is specified
        is redundant and results in a long running time when producing a report.
        We therefore find the directories containing the files which we know
        will match the file_regex ahead of time, and use this to determine 
        which directories we can prune from the search.
        """
        build_files_path = build_files_dir(self.build_dir)
        include_dirs = set()
        exclude_dirs = set()
        for dirname, _, filenames in os.walk(build_files_path):
            if any([re.search(file_regex, f) for f in filenames]):
                include_dirs.add(dirname)
            else:
                exclude_dirs.add(dirname)

        exclude_dirs.discard(str(build_files_path))
        excludes = []
        for e in exclude_dirs:
            excludes.append("--exclude-directories")
            excludes.append(e)
        return excludes


def bash(args, cwd='.', log=True, ignore_return_code=False) -> str:
    """
    Run a bash subprocess.
    This is used primarily for executing cmake, ninja, the test script and gcovr.

    Arguments
    ---
        args: Sequence or str of args, the same as you would pass to subprocess.Popen
        cwd: Path-like to the directory in which to execute the bash subprocess
        log: Bool, if True the stdout of the process is sent to stdout and the 
            result is returned as a string. Otherwise, discard the output.
        ignore_return_code: Bool, if false an erroneous return code is discarded, 
            otherwise CalledProcessError is raised.

    Returns
    ---
        str: the collected stdout of the subprocess.
    """
    process = subprocess.Popen(args,
                               cwd=cwd,
                               stdout=subprocess.PIPE,
                               universal_newlines=True)
    result = ''
    for stdout_line in iter(process.stdout.readline, ""):
        result += stdout_line
        if log:
            print(stdout_line, end='')
    process.stdout.close()
    return_code = process.wait()
    if not ignore_return_code and return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)
    else:
        return result


def run_tests(build_dir: PosixPath, test_filter_str=""):
    command = ["./test.sh", "popart", "-j16"]
    if test_filter_str:
        command += ["-R", test_filter_str]
    bash(command, build_dir, ignore_return_code=True)


def build_files_dir(build_dir: str) -> PosixPath:
    build_files_dir = Path(build_dir).joinpath(
        "build/popart/willow/CMakeFiles/popart-only.dir/src")
    return build_files_dir


def clean_coverage_output(build_dir: PosixPath):
    files_removed = 0
    for dirpath, _, filenames in os.walk(build_files_dir(build_dir)):
        for file_ in filenames:
            if file_.endswith(".gcda"):
                os.remove(os.path.join(dirpath, file_))
                files_removed += 1
    print(f"Removed {files_removed} gcov data files.")


def get_build_dir(path: Optional[str] = None):
    if path:
        build_dir = Path(path).resolve()
    else:
        try:
            build_dir = Path(os.environ['CBT_BUILDTREE'])
        except KeyError:
            print('Could not infer build directory from the CBT_BUILDTREE '
                  'environment variable.\n Either activate the build tree '
                  'or specify the build directory using the -d flag.')
            exit(1)
    return build_dir


def main():
    # Assume the base poplar view directory is two levels above this file.
    workspace_dir = Path(os.path.dirname(__file__)).resolve().parent

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        help=(
            "Optionally specify the build directory, otherwise it is inferred "
            "from the CBT_BUILDTREE environment variable."))
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help=(
            "Cleans the previous coverage output by removing all .gcda files "
            "from the build CMakeFiles."))
    parser.add_argument(
        "-p",
        "--format",
        choices=output_types.keys(),
        default="all",
        help=
        f"Specify the output format. May be one of {set(output_types.keys())}")
    parser.add_argument(
        "-o",
        "--output",
        default=workspace_dir,
        metavar='output-path',
        type=Path,
        help=
        "Optionally specify the destination of coverage report files. Defaults to the popart/ directory."
    )

    subparsers = parser.add_subparsers(dest="command")
    gcov_subparser = subparsers.add_parser("gcovr")
    gcov_subparser.add_argument(
        "-f",
        "--filter",
        metavar="gcovr-regexp",
        help=
        "Run gcovr and view coverage only for the files which match gcovr-regexp."
    )
    gcov_subparser.add_argument(
        "-e",
        "--advanced",
        metavar="gcovr-args",
        help=(
            "Run gcovr using gcovr-specific flags given as a quote-enclosed "
            "string in 'gcovr-args'. Note that this option overrides any arguments "
            "that would be invoked from the -p/--format option."))

    test_subparser = subparsers.add_parser("test")
    test_subparser.add_argument(
        "-r",
        "--ctest-regexp",
        metavar="ctest-regexp",
        default="",
        help=(
            "A Ctest -R regular expression argument against which PopART tests are "
            "matched. Only tests matching this filter are ran."))
    test_subparser.add_argument(
        "-f",
        "--filter",
        metavar="gcovr-regexp",
        default="",
        help=(
            "Regular expression specifying which files to display coverage data for. "
            "Only files matching the expression are displayed.", ))
    args = parser.parse_args()
    build_dir = get_build_dir(args.directory)

    if args.clean:
        clean_coverage_output(build_dir)
        exit(0)

    runner = GCovrRunnerParser(args.format, workspace_dir, build_dir)
    if args.output:
        runner.set_output_path(args.output)

    if args.command == "test":
        clean_coverage_output(build_dir)
        run_tests(build_dir, test_filter_str=args.ctest_regexp.strip())
        runner.set_filter(args.filter)
        report = runner.create_report()
    elif args.command == "gcovr":
        gcov_args = []
        if args.filter:
            runner.set_filter(args.filter)
        elif args.advanced:
            gcov_args = args.advanced.strip().split(" ")
        report = runner.create_report(gcov_args)
    print(report)


if __name__ == "__main__":
    main()
