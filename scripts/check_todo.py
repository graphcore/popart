#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import collections
import glob
import json
import logging
import os
import re
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
        "-d", "--debug", action="store_true", help="Print debug messages"
    )

    parser.add_argument(
        "-wf",
        "--show-wontfix",
        action="store_true",
        help="Include 'wontfix' tasks in outputs",
    )

    parser.add_argument("path", nargs="?", help="Folder to process", default=".")
    args = parser.parse_args()
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logging_level)
    logger.debug(f"Args: {str(args)}")
    os.chdir(args.path)

    # Find all the TODOs in the code base
    tasks = collections.defaultdict(list)
    total_malformed = 0

    for f in glob.glob("./**", recursive=True):
        if (
            os.path.isdir(f)
            or "__pycache__" in f
            or os.path.basename(f) in FILES_TO_SKIP
        ):
            continue

        try:
            qualified_filename = os.path.normpath(os.path.join(args.path, f))
            for line, text in enumerate(open(f, "r", encoding="utf-8"), start=1):
                location = f"{qualified_filename}:{line}"

                match_todo = re.match(r".*TODO", text)
                match_task = re.match(r".*[\W]+(T\d+)", text)

                task_id = (
                    match_task.group(1)
                    if match_task and len(match_task.group(1)) >= 5
                    else None
                )

                if match_todo and task_id:
                    logger.debug(f"Found {task_id} in {location}")
                    tasks[task_id].append(location)

                if task_id and not match_todo:
                    logger.warning(f"Task ID without 'TODO' {location} {text.rstrip()}")
                    tasks[task_id].append(location)

                if match_todo and not task_id:
                    total_malformed += 1
                    logger.warning(
                        f"TODO missing task number {location} {text.rstrip()}"
                    )

        except UnicodeDecodeError:
            logger.debug(f"Ignoring binary file: {f}")

    total_could_remove = 0
    for task, locations in tasks.items():
        title, status, is_closed = get_task_info(task)
        if is_closed and (status.upper() != "WONTFIX" or args.show_wontfix):
            total_could_remove += len(locations)
            locations_message = "\n\t".join(locations)
            logger.warning(
                f"[{task}: {title}] is closed ({status.upper()}), following TODOs need"
                f" to be removed/updated:\n\t{locations_message}"
            )
        else:
            logger.debug(f"{task}: is still open: OK")

    total_todo = sum([len(locations) for task, locations in tasks.items()])
    logger.info(f"Total TODOs with Task IDs (open & closed) = {total_todo}")
    logger.info(f"Total TODOs without Task IDs = {total_malformed}")
    logger.info(f"Total TODOs which could be removed = {total_could_remove}")
