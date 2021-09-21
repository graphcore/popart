# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart._internal.ir as _ir

__all__ = ['contains_op_of_type']


def contains_op_of_type(opType: str, PbOpType, graph: pir.Graph) -> bool:
    pb_g = graph._pb_graph

    for pb_op in pb_g.getOps():
        if pb_op.opType() == opType and isinstance(pb_op, PbOpType):
            return True
    return False
