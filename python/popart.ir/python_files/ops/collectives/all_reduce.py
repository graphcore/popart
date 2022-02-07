# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, List
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from popart.ir.errors import UndefinedValue
from ..utils import check_in_graph
from .collectives import to_collective_op, CollectiveOps


@op_debug_context
def all_reduce(
        ts: List[Tensor],
        ipus: Optional[List[int]] = None,
        op: CollectiveOps = 'add',
) -> List[Tensor]:
    """
    All-reduce tensors across IPUs within a replica.

    Currently only the 'add' reduce op is supported by autodiff.

    Args:
        ts (List[Tensor]): Tensors to reduce
        ipus (Optional[List[int]]): IPUs the tensors are located on. If None the op will try and infer.
        op (str): The reducing op. Options: add, mean, mul, min, max, and, or, square_add, local.

    Returns:
        List[Tensor]: Output Tensors. Each Tensors data is identical on a IPU corresponding to `ipus`
    """
    return _all_reduce(ts, ipus, op)


@op_debug_context
def all_reduce_identical_inputs(
        ts: List[Tensor],
        ipus: Optional[List[int]] = None,
        op: CollectiveOps = 'add',
) -> List[Tensor]:
    """
    All-reduce tensors across IPUs within a replica where the input tensors are identical.
    This means the op is an identity but the corresponding grad op is an all-reduce.

    Currently only the 'add' reduce op is supported by autodiff.

    The `AllReduceToIdentityPattern` pattern must be run for this op to function correctly.

    Args:
        ts (List[Tensor]): Tensors to reduce
        ipus (Optional[List[int]]): IPUs the tensors are located on. If None the op will try and infer.
        op (str): The reducing op. Options: add, mean, mul, min, max, and, or, square_add, local.

    Returns:
        List[Tensor]: Output Tensors. Each Tensors data is identical on a IPU corresponding to `ipus`

    """
    return _all_reduce(ts, ipus, op, identical_inputs=True)


@op_debug_context
def all_reduce_identical_grad_inputs(
        ts: List[Tensor],
        ipus: Optional[List[int]] = None,
        op: CollectiveOps = 'add',
) -> List[Tensor]:
    """
    All-reduce tensors across IPUs within a replica where the grad tensors of the corresponding grad op are identical.
    This means that this op is an all-reduce and the corresponding grad op an identity.

    Currently only the 'add' reduce op is supported by autodiff.

    The `AllReduceToIdentityPattern` pattern must be run for this op to function correctly.

    Args:
        ts (List[Tensor]): Tensors to reduce
        ipus (Optional[List[int]]): IPUs the tensors are located on. If None the op will try and infer.
        op (str): The reducing op. Options: add, mean, mul, min, max, and, or, square_add, local.

    Returns:
        List[Tensor]: Output Tensors. Each Tensors data is identical on a IPU corresponding to `ipus`

    """
    return _all_reduce(ts, ipus, op, identical_grad_inputs=True)


def _all_reduce(
        ts: List[Tensor],
        ipus: Optional[List[int]],
        op: CollectiveOps,
        identical_inputs: bool = False,
        identical_grad_inputs: bool = False,
) -> List[Tensor]:
    op = to_collective_op(op)

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, **{f't{i}': t for i, t in enumerate(ts)})

    if ipus is None:
        try:
            ipus = [t.ipu for t in ts]
        except UndefinedValue as e:
            raise ValueError(
                "Could not automatically infer the IPU of all input Tensors `ts`. "
                "Please specify the IPUs via the `ipus` parameter.") from e

    if len(ts) != len(ipus):
        raise ValueError(
            f"Number of specified tensor does not equal number of specified IPUs. "
            f"{len(ts)} != {len(ipus)}")

    opid = _ir.OperatorIdentifier("ai.graphcore", "AllReduce", 1,
                                  _ir.NumInputs(2, 2), 1)
    settings = ctx._get_op_settings("all_reduce")

    op = pb_g.createConnectedOp_AllReduceOp(
        {i: t.id
         for i, t in enumerate(ts)}, {
             i: g._create_tensor_id(f"all_reduce_out_{i}")
             for i in range(len(ts))
         },
        opid,
        op,
        ipus,
        identicalInputs_=identical_inputs,
        identicalGradInputs_=identical_grad_inputs,
        settings=settings)

    output = [Tensor._from_pb_tensor(op.outTensor(i)) for i in range(len(ts))]

    return output
