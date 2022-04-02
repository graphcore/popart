# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph


@op_debug_context
def transpose(t: Tensor,
              permutation: Optional[Tuple[int, ...]] = None) -> Tensor:
    """
    Permute the axes of a tensor.

    By default this operation reverses the axes of `t`.

    This is similar to :onnxop:`Transpose`.

    Args:
        t (Tensor): Tensor to be transposed.
        permutation (Optional[Iterable[int]]): Iterable containing the permutation of [0, N-1] where N is the
        rank of input `t`. If not provided, the axes will be reversed.
    Returns:
        out (Tensor): The transposed tensor.
    """
    permutation = _handle_permutation(t, permutation)

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
    Permute the axes of a tensor (in-place).

    By default this operation reverses the axes of `t`.

    This is the in-place version of :func:`~ops.transpose`. The behaviour is the same, but it modifies the
    tensor in place.

    This is similar to :onnxop:`Transpose`.

    Args:
        t (Tensor): Tensor to be transposed.
        permutation (Optional[Tuple[int, ...]]):
            Tuple containing the a permutation of [0, N-1] where N is the
            rank of input `t`. If not provided, the axes will be reversed.
    Returns:
        out (Tensor):
            The transposed input tensor.
    """
    permutation = _handle_permutation(t, permutation)

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


def _handle_permutation(
        t: Tensor, permutation: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Check if the values of a permutation is valid given a tensor.

    The default is to use the reversed dimensions as a permutation (if None is given).

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
