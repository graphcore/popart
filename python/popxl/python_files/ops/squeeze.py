# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional, List
from popxl.context import debug_context_frame_offset
from popxl.tensor import Tensor
from .reshape import reshape
from .utils import handle_negative_axis


@debug_context_frame_offset(1)
def squeeze(t: Tensor, axes: Optional[List[int]] = None) -> Tensor:
    """
    Remove axes of length one from the tensor.

    Takes an input `axes` with a list of axes to squeeze.
    If `axes` is not provided, all the single dimensions will be removed from the shape.
    If an axis is selected with shape entry not equal to one, an error is raised.
    Implemented using `reshape` under the hood.

    Args:
        t: Tensor
            Tensor to be squeezed.
        axes: List[int]
            List of integers indicating the dimensions to squeeze.
            Negative value means counting dimensions from the back.
            Accepted range is `[-r, r-1]` where `r = rank(t)`.
    Returns:
        out: Tensor
            Squeezed tensor.
    """

    shape = t.shape

    if axes is None:
        shape_new = [e for e in shape if e != 1]
    else:
        axes = sorted(handle_negative_axis(t, e) for e in axes)

        if len(axes) > len(set(axes)):
            raise ValueError("Axes contains duplicates")

        shape_new = list(shape[:])
        for i in axes[::-1]:
            if i > len(shape_new) - 1:
                raise ValueError(
                    f"Can't squeeze axis '{i}' as tensor has rank '{len(shape_new)}'"
                )
            if shape[i] != 1:
                raise ValueError(
                    f"Can't squeeze axis '{i}' as length of axis is >1: {shape[i]}"
                )

            del shape_new[i]

    t = reshape(t, shape=shape_new)
    return t
