# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path, PosixPath

from elasticsearch import RequestsHttpConnection
from elasticsearch_dsl import Date, Document, Float, Keyword, connections
from view_coverage import GCovrRunnerParser, clean_coverage_output, run_tests


class CoverageMetrics(Document):
    """
    Unit test statement and branch coverage across all source
    files in PopART
    """
    unittest_branch_covered = Float(required=True)
    unittest_branch_total = Float(required=True)
    unittest_branch_percent = Float(required=True)
    unittest_line_covered = Float(required=True)
    unittest_line_total = Float(required=True)
    unittest_line_percent = Float(required=True)
    timestamp = Date(required=True, default_timezone='UTC')
    diff_id = Keyword()

    class Index:
        name = "popart_coverage"

    def save(self, **kwargs):
        self.timestamp = datetime.utcnow()
        return super().save(**kwargs)


def connect_to_server(server: str, cookie: str):
    if "_oauth2_proxy" not in cookie:
        cookie = "_oauth2_proxy=" + cookie
    connections.create_connection(
        hosts=[server],
        headers={"cookie": cookie} if cookie else None,
        connection_class=RequestsHttpConnection)


def get_unittest_coverage_json_report(workspace_dir: PosixPath,
                                      build_dir: PosixPath) -> dict:
    clean_coverage_output(build_dir)
    run_tests(build_dir, test_filter_str="unittest")
    runner = GCovrRunnerParser(output="json",
                               workspace_dir=workspace_dir,
                               build_dir=build_dir)
    return json.loads(runner.create_report())


def get_diff_id(workspace_dir: PosixPath) -> str:
    """Extract the differential revision ID either from the environment
    or the commit message of the latest diff.

    If the revision ID is not present in the environment, we execute
    git show --summary from `workspace_dir` and parse this for the
    diff ID. We therefore expect the most recent commit to be one from
    an arc patch.

    If we cannot determine a diff ID we return the empty string.
    """
    diff_revision_var = os.environ.get("GCCI_DIFF_REVISION")
    if diff_revision_var:
        return diff_revision_var
    process = subprocess.Popen(["git", "show", "--summary"],
                               cwd=workspace_dir,
                               stdout=subprocess.PIPE,
                               universal_newlines=True)
    stdout, _ = process.communicate()
    # There is probably a more programatic way to find the Diff Id
    # of the revision being built, but using the conduit API is a headache.
    index = stdout.find("Reviewers:")
    match = re.findall(
        r'Differential Revision: https://phabricator.sourcevertex.net/(?P<diff_id>D\d+)',
        stdout[index:])
    if match:
        return match[-1]
    return ''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "build_dir",
        type=Path,
        help=
        "Path to the directory contaning the popart build e.g build/build/popart"
    )
    parser.add_argument(
        "results_server",
        type=str,
        help="URL of the elasticsearch cluster to which results are uploaded")
    parser.add_argument("--cookie-file",
                        type=Path,
                        help="Path to cookie file.")

    args = parser.parse_args()

    with open(args.cookie_file) as fp:
        cookie = fp.read().strip()

    # Provided this script stays where it is, this is a safe bet
    workspace_dir = Path(os.path.dirname(__file__)).resolve().parent
    # We have to determine where the poplar build directory is from the
    # CMake CMAKE_BINARY_DIR variable (passed as a command line arg),
    # but this variable points to the poplar build subdirectory, so we
    # have to go two levels up.
    poplar_build_dir = args.build_dir.resolve().parent.parent
    report = get_unittest_coverage_json_report(workspace_dir, poplar_build_dir)
    server = args.results_server
    connect_to_server(server, cookie)

    if not CoverageMetrics._index.exists():
        CoverageMetrics.init()

    metrics = CoverageMetrics()
    metrics.unittest_branch_covered = report['branch_covered']
    metrics.unittest_branch_total = report['branch_total']
    metrics.unittest_branch_percent = report['branch_percent']
    metrics.unittest_line_covered = report['line_covered']
    metrics.unittest_line_total = report['line_total']
    metrics.unittest_line_percent = report['line_percent']
    metrics.diff_id = get_diff_id(workspace_dir)
    metrics.save()
