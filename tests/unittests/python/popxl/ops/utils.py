# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Type

import popxl
import popart._internal.ir as _ir


def contains_op_of_type(opType: str, PbOpType: Type[_ir.Op],
                        graph: popxl.Graph) -> bool:
    pb_g = graph._pb_graph

    for pb_op in pb_g.getOps():
        if pb_op.opType() == opType and isinstance(pb_op, PbOpType):
            return True
    return False


def num_op_of_type(opType: str, PbOpType: Type[_ir.Op],
                   graph: popxl.Graph) -> int:
    pb_g = graph._pb_graph

    count = 0

    for pb_op in pb_g.getOps():
        if pb_op.opType() == opType and isinstance(pb_op, PbOpType):
            count += 1

    return count
