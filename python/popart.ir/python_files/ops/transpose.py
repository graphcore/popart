# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def transpose(t: Tensor,
              permutation: Optional[Tuple[int, ...]] = None) -> Tensor:
    """
    Permute the axes of a Tensor. By default reverses the axes of t.

    Args:
        t: Tensor
            Tensor to be transposed.
        permutation: tuple of ints (optional)
            Tuple containing the a permutation of [0, N-1] where N is the
            rank of input `t`. If not provided, the axes will be reversed.
    Returns:
        out: Tensor
            The transposed tensor
    """
    permutation = _handle_permuation(t, permutation)

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('transpose')
    opid = _ir.OperatorIdentifier("ai.onnx", "Transpose", 1, _ir.NumInputs(
        1, 1), 1)
    op = pb_g.createConnectedOp_TransposeOp(
        {0: t.id},
        {0: g._create_tensor_id(f"{t.name}_T")},
        opid,
        permutation,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def transpose_(t: Tensor,
               permutation: Optional[Tuple[int, ...]] = None) -> Tensor:
    """
    Permute the axes of a Tensor. By default reverses the axes of t.

    This is the inplace version of :func:`~ops.transpose`. Behaviour is the same, but modifies the
        tensor inplace.

    Args:
        t: Tensor
            Tensor to be transposed.
        permutation: tuple of ints (optional)
            Tuple containing the a permutation of [0, N-1] where N is the
            rank of input `t`. If not provided, the axes will be reversed.
    Returns:
        out: Tensor
            The transposed tensor
    """
    permutation = _handle_permuation(t, permutation)

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    settings = ctx._get_op_settings('transpose_inplace')
    opid = _ir.OperatorIdentifier("ai.graphcore", "TransposeInplace", 1,
                                  _ir.NumInputs(1, 1), 1)
    op = pb_g.createConnectedOp_TransposeInplaceOp(
        {0: t.id},
        {0: g._create_tensor_id(f"{t.name}_T")},
        opid,
        permutation,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def _handle_permuation(
        t: Tensor, permutation: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Give a default (reversed dimensions) permutation if None is provided,
        and check if values are valid.

    Args:
        t (Tensor): The tensor to be permuted.
        permutation (Optional[Tuple[int, ...]]): The permutation to use.

    Raises:
        ValueError: If the values in the permutation are greater than the tensor's rank.

    Returns:
        Tuple[int, ...]: The permutation to use.
    """
    if permutation is None:
        # Reverse dimensions
        permutation = tuple(reversed(range(t.rank)))

    if any(map(lambda dim: dim >= t.rank, permutation)):
        raise ValueError(
            f"Values in permutation must be less than the tensor's rank {t.rank}. "
            f"Found {tuple(filter(lambda dim: dim >= t.rank, permutation))}")

    return permutation
