# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import List

import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor

from .utils import check_in_graph


@op_debug_context
def histogram(t: Tensor, levels: List[float], absolute_of_input: bool) -> Tensor:
    r"""
    Compute the histogram of the input tensor.

    All but the last bin are half-open. In other words, if `levels` is:

    ```
    [1, 2, 3, 4]
    ```

    then the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3). The last bin, however, is [3, 4], which includes 4.

    See also `PyTorch torch.histc <https://pytorch.org/docs/stable/generated/torch.histc.html>`__, :py:func:`NumPy histogram <numpy.histogram>`.

    Args:
        t:
            Input tensor.
        levels:
            A monotonically increasing list of bin edges.
        absolute_of_input:
            If True, the absolute value of each input is calculated before comparison to the `levels` data.

    Returns:
        Tensor:
            Counts of the number of values in each bin.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    check_increment(levels)

    settings = ctx._get_op_settings("histogram")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "Histogram", 1, _ir.NumInputs(1, 1), 1
    )
    op = pb_g.createConnectedOp_HistogramOp(
        {0: t.id},
        {0: g._create_tensor_id("histogram_out")},
        opid,
        levels_=levels,
        absoluteOfInput_=absolute_of_input,
        settings=settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))


def check_increment(levels: List[float]):
    """
    Check if the inputs are monotonically increasing.
    """
    ascend_list = all([levels[i] < levels[i + 1] for i in range(len(levels) - 1)])
    if not ascend_list:
        raise ValueError(
            f"The input levels `{levels}` is not a monotonically increasing list."
        )
