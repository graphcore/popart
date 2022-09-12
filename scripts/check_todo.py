#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import glob
import os
import re
import collections
import logging
import json
import subprocess
from typing import Dict, Tuple

logger = logging.getLogger(os.path.basename(__file__))

FILES_TO_SKIP = [os.path.basename(__file__), "pylintrc", "Doxyfile.in"]


def arc_call_conduit(command: str, stdin: object) -> Dict[str, object]:
    """Run a command to arcanist, and return the response.

    Args:
        command: The command to run.
        stdin: The

    Raises:
        Exception: If the command was unsuccessful.
        Exception: If the response from phabricator was an error.

    Return: The response from the command.
    """
    try:
        output = subprocess.check_output(
            [
                "arc",
                "--config",
                "phabricator.uri=https://phabricator.sourcevertex.net/",
                "call-conduit",
                command,
                "--",
            ],
            input=json.dumps(stdin),
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as e:
        # Only keep the error summary if it's available
        errors = e.stderr or e.stdout
        raise Exception(f"arc call-conduit {command} Failed:\n{errors}")

    reply = json.loads(output)
    if reply["error"]:
        raise Exception(
            f"ERROR: Failed to talk to Phabricator: Command {command}, stdin"
            f" {json.dumps(stdin)}, failed: {reply['errorMessage']}"
        )
    return reply["response"]


def get_task_info(task: str) -> Tuple[str, str, bool]:
    """Get information about a phabricator task.

    Args:
        task: The "T1234" number to lookup.

    Return: The title, status and whether the task is closed.
    """
    assert task[0] == "T", "Expected task ID of form: T1234"

    reply = arc_call_conduit("maniphest.info", {"task_id": int(task[1:])})
    return str(reply["title"]), str(reply["status"]), bool(reply["isClosed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Print debug messages"
    )

    parser.add_argument("path", nargs="?", help="Folder to process")
    args = parser.parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)
    logger.debug("Args: %s", str(args))

    expr = re.compile(r".*TODO.*(T\d+)")
    expr_loose = re.compile(r".*TODO.*")

    # Find all the TODOs in the code base
    tasks = collections.defaultdict(list)
    total_malformed = 0
    if args.path:
        os.chdir(args.path)

    for f in glob.glob("./**", recursive=True):
        if os.path.isdir(f) or os.path.basename(f) in FILES_TO_SKIP:
            continue

        try:
            for line, txt in enumerate(open(f, "r", encoding="utf-8"), start=1):
                m = expr.match(txt)

                if m:
                    location = f"{f}:{line}"
                    logger.debug("Found %s in %s", m.group(1), location)
                    tasks[m.group(1)].append(location)
                elif expr_loose.match(txt):
                    total_malformed += 1
                    logger.warning(
                        f"TODO missing task number {f}:{line} {txt.rstrip()}"
                    )

        except UnicodeDecodeError:
            logger.debug("Ignoring binary file %s", f)

    total_could_remove = 0
    for task, locations in tasks.items():
        title, status, is_closed = get_task_info(task)
        if is_closed:
            total_could_remove += len(locations)
            locations_message = "\n\t".join(locations)
            logger.warning(
                f"[{task}: {title}] is closed ({status.upper()}), following TODOs need"
                f" to be removed/updated:\n\t{locations_message}"
            )
        else:
            logger.debug("%s: is still open: OK", task)

    total_todo = sum([len(locations) for task, locations in tasks.items()])
    logger.info(f"Total TODOs with Task IDs (open & closed) = {total_todo}")
    logger.info(f"Total TODOs without Task IDs = {total_malformed}")
    logger.info(f"Total TODOs which could be removed = {total_could_remove}")
