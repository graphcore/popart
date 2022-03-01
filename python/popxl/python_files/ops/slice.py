# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, Union, List
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph

INT_MIN = -2**31
INT_MAX = 2**31 - 1


@op_debug_context
def slice(t: Tensor,
          start: Optional[Union[int, List[Optional[int]]]] = None,
          stop: Optional[Union[int, List[Optional[int]]]] = None,
          step: Optional[Union[int, List[Optional[int]]]] = None,
          axis: Optional[Union[int, List[int]]] = None) -> Tensor:
    """
    Selects elements from a tensor using a slice or multiple slices.

    A slice specifies the start (inclusive) and stop (exclusive) index of elements to select.
    Multiple slices can be specified using a list of items for each parameter (start, stop, step).
    If step is `-1` the slice is performed backwards.

    If axis is not specified, each slice will correspond to axis 0 to N where N is the number of slices.

    Examples:

    .. code-block:: python

        t == slice(t) == slice(t, axis=1)
        slice(t, start=1)           # Slice axis 0 from start index 1
        slice(t, start=[1,2]) == slice(t, start=[1,2], axis=[0,1])
        slice(t, stop=-2)           # Slice axis 0 upto second last element (exclusive)
        slice(t, stop=3, step=-1)   # Slice backwards from last element (inclusive) to third last element (exclusive)

    Args:
        t (Tensor): Tensor to slice
        start: Index of first element (inclusive) or `None` which defaults to 0.
        stop: Index of last element (exclusive) or `None` which defaults to last
            element (inclusive) if step is forward or first element (inclusive) if step is backwards.
        step: `1` for forward or `-1` for backwards.
        axis: Axis of tensor to slice on or `None` will default to each axis sequentially.

    Returns:
        Tensor: output tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if start is None and stop is None and step is None:
        return t

    start, stop, step, axis = process_args(start, stop, step, axis)

    opid = _ir.OperatorIdentifier("ai.onnx", "Slice", 11, _ir.NumInputs(1, 1),
                                  1)
    settings = ctx._get_op_settings("slice")
    op = pb_g.createConnectedOp_SliceOp(
        {0: t.id},
        {0: g._create_tensor_id("slice_out")},
        starts_=start,
        ends_=stop,
        axes_=axis,
        steps_=step,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def slice_(t: Tensor,
           start: Optional[Union[int, List[Optional[int]]]] = None,
           stop: Optional[Union[int, List[Optional[int]]]] = None,
           step: Optional[Union[int, List[Optional[int]]]] = None,
           axis: Optional[Union[int, List[int]]] = None) -> Tensor:
    """
    Selects elements from a tensor using a slice or multiple slices (inplace).

    This is the inplace version of :func:`~ops.slice`. Behaviour is the same, but modifies the
        tensor inplace.

    Args:
        t (Tensor): Tensor to slice
        start: Index of first element (inclusive) or `None` which defaults to 0.
        stop: Index of last element (exclusive) or `None` which defaults to last
            element (inclusive) if step is forward or first element (inclusive) if step is backwards.
        step: `1` for forward or `-1` for backwards.
        axis: Axis of tensor to slice on or `None` will default to each axis sequentially.

    Returns:
        Tensor: alias of the input tensor t.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    if start is None and stop is None and step is None:
        return t

    start, stop, step, axis = process_args(start, stop, step, axis)

    opid = _ir.OperatorIdentifier("ai.graphcore", "SliceInplace", 1,
                                  _ir.NumInputs(1, 1), 1)
    settings = ctx._get_op_settings("slice_inplace")
    op = pb_g.createConnectedOp_SliceInplaceOp(
        {0: t.id},
        {0: g._create_tensor_id("slice_out")},
        starts_=start,
        ends_=stop,
        axes_=axis,
        steps_=step,
        opid=opid,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def process_args(start: Optional[Union[int, List[Optional[int]]]] = None,
                 stop: Optional[Union[int, List[Optional[int]]]] = None,
                 step: Optional[Union[int, List[Optional[int]]]] = None,
                 axis: Optional[Union[int, List[int]]] = None
                 ) -> Tuple[List[int], List[int], List[int], List[int]]:

    # Convert to list if scalar
    start = [start] if start is not None and isinstance(start, int) else start
    stop = [stop] if stop is not None and isinstance(stop, int) else stop
    step = [step] if step is not None and isinstance(step, int) else step
    axis = [axis] if axis is not None and isinstance(axis, int) else axis

    # Check lengths
    N = None
    kw = dict(start=start, stop=stop, step=step, axis=axis)
    for k, v in kw.items():
        if v is not None:
            if N is None:
                N = len(v)
            elif N != len(v):
                raise ValueError(
                    f"All inputs must have same length. `{k}` has length {len(v)} != {N}."
                )

    # Convert to default if `None` or element is `None`
    if step is None:
        step = [1] * N
    else:
        step = [(1 if e is None else e) for e in step]

    axis = list(range(N)) if axis is None else axis

    if start is None:
        start = [(0 if step_i > 0 else -1) for step_i in step]
    else:
        start = [((0 if step[i] > 0 else -1) if e is None else e)
                 for i, e in enumerate(start)]

    if stop is None:
        stop = [(INT_MAX if step_i > 0 else INT_MIN) for step_i in step]
    else:
        stop = [((INT_MAX if step[i] > 0 else INT_MIN) if e is None else e)
                for i, e in enumerate(stop)]

    return start, stop, step, axis
