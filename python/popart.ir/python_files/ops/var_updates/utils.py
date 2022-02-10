# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Any, Dict
import popart._internal.ir as _ir
from popart.ir.tensor import Tensor


def handle_optimizer_value(f: Any, ins: Dict[int, str],
                           index: int) -> _ir.OptimizerValue:
    if isinstance(f, Tensor):
        ins[index] = f.id
        return _ir.OptimizerValue(0.0, False)
    elif f is None:
        return _ir.OptimizerValue()

    return _ir.OptimizerValue(f, True)
