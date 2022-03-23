# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import json
import os
from datetime import datetime
from pathlib import Path, PosixPath

from elasticsearch import RequestsHttpConnection
from elasticsearch_dsl import Boolean, Date, Document, Float, Keyword, connections
from view_coverage import GCovrRunnerParser, clean_coverage_output, run_tests


class CoverageMetrics(Document):
    """
    Unit test statement and branch coverage across all source files in PopART.
    """
    code_metrics_type = Keyword(required=True)
    unittest_branch_covered = Float(required=True)
    unittest_branch_total = Float(required=True)
    unittest_branch_percent = Float(required=True)
    unittest_line_covered = Float(required=True)
    unittest_line_total = Float(required=True)
    unittest_line_percent = Float(required=True)
    timestamp = Date(required=True, default_timezone='UTC')
    is_diff_build = Boolean(required=True)
    diff_id = Keyword(required=False)
    commit_id = Keyword(required=False)

    class Index:
        name = "code_metrics"

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


def set_build_details(metrics: CoverageMetrics) -> str:
    """ Set is_diff_build, diff_id and commit_id as appropriate. """

    is_diff_build = (os.environ.get("GCCI_DIFF_BUILD", "false") == "true")

    if not is_diff_build:
        metrics.is_diff_build = False
        metrics.commit_id = os.environ.get("GIT_COMMIT", "<undefined>")

    else:
        metrics.is_diff_build = True
        metrics.diff_id = os.environ.get("GCCI_DIFF_REVISION", "<undefined>")


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
    metrics.code_metrics_type = "PopART Test Coverage"
    metrics.unittest_branch_covered = report['branch_covered']
    metrics.unittest_branch_total = report['branch_total']
    metrics.unittest_branch_percent = report['branch_percent']
    metrics.unittest_line_covered = report['line_covered']
    metrics.unittest_line_total = report['line_total']
    metrics.unittest_line_percent = report['line_percent']
    set_build_details(metrics)
    metrics.save()
