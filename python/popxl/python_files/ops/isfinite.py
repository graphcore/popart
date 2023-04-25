# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from popxl.context import op_debug_context
from popxl.tensor import Tensor

from .identity import rename
from .isnan import isnan
from .isinf import isinf


@op_debug_context
def isfinite(t: Tensor) -> Tensor:
    """
    Return a boolean tensor of the same shape indicating which elements are finite (not NaN or infinity).

    Args:
        t (Tensor):
            Tensor to check.

    Returns:
        Tensor: boolean tensor of the same shape indicating which elements are finite (not NaN or infinity).
    """
    out = ~(isnan(t) | isinf(t))
    out = rename(out, "isfinite_out")
    return out
