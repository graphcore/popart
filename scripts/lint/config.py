# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

__all__ = ['LinterConfig', 'load_linter_configs']


@dataclass
class LinterConfig:
    name: str
    class_: str
    version: Optional[str] = None
    exclude: Optional[Union[str, List[str]]] = None
    include: Optional[Union[str, List[str]]] = None
    config_file: Optional[Path] = None


def load_linter_configs(path_to_config: str) -> dict:
    assert os.path.isfile(
        path_to_config), f"Could not find linter config file: {path_to_config}"

    with open(path_to_config, "r", encoding="utf-8") as fp:
        configs = json.load(fp)

    if 'linters' not in configs:
        raise KeyError(
            "The linter config file must contain a root 'linters' field.")
    return configs
