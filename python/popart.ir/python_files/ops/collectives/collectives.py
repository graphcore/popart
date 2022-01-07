# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from typing import Union
from typing_extensions import Literal

__all__ = ['CommGroup', 'CommGroupType']

CommGroup = _ir.CommGroup
CommGroupType = _ir.CommGroupType

OP_MAP = {
    'add': _ir.CollectiveOperator.Add,
    'mean': _ir.CollectiveOperator.Mean,
    'mul': _ir.CollectiveOperator.Mean,
    'min': _ir.CollectiveOperator.Min,
    'max': _ir.CollectiveOperator.Max,
    'and': _ir.CollectiveOperator.LogicalAnd,
    'or': _ir.CollectiveOperator.LogicalOr,
    'square_add': _ir.CollectiveOperator.SquareAdd,
    'local': _ir.CollectiveOperator.Local,
}

CollectiveOps = Literal['add', 'mean', 'mul', 'min', 'max', 'and', 'or',
                        'square_add', 'local']


def to_collective_op(op: CollectiveOps) -> _ir.CollectiveOperator:
    try:
        return OP_MAP[op]
    except KeyError:
        raise ValueError(f"Not a valid op: {op}. "
                         f"Must choose from: {', '.join(OP_MAP.keys())}")
