# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
import sys
from collections import namedtuple
from importlib import import_module
from pathlib import Path
from typing import List, Tuple

from lint import LinterConfig, load_linter_configs
from util import get_project_source_dir

LinterMessage = namedtuple(typename='LinterMessage',
                           field_names=['linter', 'message'])


class LinterOutput:
    def __init__(self,
                 status_ok: bool,
                 messages: List[LinterMessage],
                 original: str = '',
                 replacement: str = '') -> None:
        self.status_ok = status_ok
        self.original = original
        self.replacement = replacement
        self.messages: List[LinterMessage] = messages

    def json(self):
        return json.dumps({
            "status_ok": self.status_ok,
            "original": self.original,
            "replacement": self.replacement,
            "messages": [msg._asdict() for msg in self.messages]
        })


def lint(file_to_lint: str, linter_config_file: str):
    configs = create_linter_configs(linter_config_file)
    linters, error_msgs = create_linters_and_check_for_errors(configs)
    if error_msgs:
        return LinterOutput(status_ok=False, messages=error_msgs)
    return run_linters(file_to_lint, linters)


def create_linter_configs(config_file: str) -> List[LinterConfig]:
    configs = load_linter_configs(config_file)

    config_objects = []
    for linter_name, config_dict in configs['linters'].items():
        if 'class' not in config_dict:
            raise KeyError(
                f"Config for linter {linter_name} is missing the required 'class' field."
            )

        config_file = config_dict.get('config_file')
        path_to_config = '' if not config_file else str(
            get_project_source_dir().joinpath(config_file))
        config_objects.append(
            LinterConfig(name=linter_name,
                         class_=config_dict['class'],
                         version=config_dict.get('version'),
                         include=config_dict.get('include'),
                         exclude=config_dict.get('exclude'),
                         config_file=path_to_config))
    return config_objects


def create_linters_and_check_for_errors(
        configs: List[LinterConfig]) -> Tuple[list, List[LinterMessage]]:
    error_msgs = []
    linters = []
    for config in configs:
        linter = linter_from_config(config)
        error_msg = _error_message_if_linter_invalid(linter, config)
        if error_msg:
            error_msgs.append(error_msg)
        else:
            linters.append(linter)
    return linters, error_msgs


def _error_message_if_linter_invalid(linter, config) -> str:
    error_msg = None
    if not linter.is_available():
        error_msg = LinterMessage(
            linter.name,
            f"Linter {linter.name} is not available or installed. "
            f"Install using: {linter.install_instructions(config.version)}")
    elif config.version and not _correct_version(linter, config):
        error_msg = LinterMessage(
            linter.name,
            f"Version requirement not satisfied for linter {linter.name}. "
            f"Version required: {config.version}. "
            f"You have: {('.'.join(str(i) for i in linter.get_version())) if linter.get_version() != None else None}"
        )

    return error_msg


def _correct_version(linter, config: LinterConfig) -> bool:
    required_version = tuple(
        int(i) if i.isnumeric() else i for i in config.version.split('.'))
    available_version = linter.get_version()
    return available_version == required_version


def linter_from_config(linter_config: LinterConfig):
    linter_class = import_from_string(linter_config.class_)
    linter = linter_class(linter_config)
    return linter


def import_from_string(module_path: str):
    try:
        module_path, class_name = module_path.rsplit('.', 1)
    except ValueError as err:
        raise ImportError(
            f"{module_path} doesn't look like a valid linter module path"
        ) from err

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError as err:
        raise ImportError(
            f'Module "{module_path}" does not define a "{class_name}" linter'
        ) from err


def run_linters(file_to_lint, linters) -> LinterOutput:
    with open(file_to_lint, 'r') as fp:
        original = fp.read()

    linter_messages = []
    file_contents = original
    for linter in linters:
        after = linter.apply(file_to_lint, file_contents)
        if file_contents != after:
            linter_messages.append(
                LinterMessage(linter.name, linter.get_linter_message()))
            file_contents = after
    return LinterOutput(status_ok=True,
                        original=original,
                        replacement=file_contents,
                        messages=linter_messages)


if __name__ == "__main__":
    config_file = Path(__file__).resolve().parent.joinpath(
        "lint/linter_config.json")
    output = lint(file_to_lint=sys.argv[1], linter_config_file=config_file)
    print(output.json())
