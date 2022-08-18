# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from typing_extensions import Literal
from .utils import check_in_graph

PrintTensorFmt = _ir.PrintTensorFmt
FloatFormat = _ir.FloatFormat


FLOAT_FORMAT_MAP = {
    "auto": _ir.FloatFormat.Auto,
    "fixed": _ir.FloatFormat.Fixed,
    "scientific": _ir.FloatFormat.Scientific,
    "none": _ir.FloatFormat.None_,
}

FloatFormat = Literal["auto", "fixed", "scientific", "none"]


@op_debug_context
def print_tensor(
    t: Tensor,
    title: str = None,
    print_self: bool = True,
    print_gradient: bool = False,
    summarise_threshold: int = 1000,
    edge_items: int = 3,
    max_line_width: int = 75,
    digits: int = 8,
    float_format: FloatFormat = "auto",
    separator: str = " ",
    open_bracket: str = "[",
    close_bracket: str = "]",
) -> Tensor:
    """
    Print a tensor.

    The output tensor of this op must be consumed if you want to print the gradient tensor.
    If the output is not consumed this op does not get pruned when running `removeIsolatedTensors`.

    The default output format will split large lines, print all elements in the
    same format, pad elements so that they align and summarise large tensors.

    Args:
        t (Tensor): The tensor to print.
        title (str, optional): Title to print. Defaults to None.
        print_self (bool, optional): Print the tensor itself. Defaults to `True`.
        print_gradient (bool, optional): Indicates if the associated gradient tensor
            of t is also printed (`True`) or not (`False`). Defaults to False.
        summarise_threshold (int): default 1000. If the number of elements of
            the tensor exceeds this threshold the output will be summarised.
            Only the edge elements will be displayed with an ellipsis
            indicating skipped elements. A value of 0 will disable
            summarisation.
        edge_items (int): default 3. Number of edge elements to include at the
            beginning and end when summarisation is enabled.
        max_line_width (int): default 75. lines longer than this limit will
            be split across multiple lines. A value of 0 will disable line
            splitting.
        digits (int): default 8. Number of digits to display. For integers this
            limit can be exceeded if any number is large enough. For floating
            points this does not include the exponent. The number of digits is
            used in conjunction analysis of the tensor to determine the width
            of each element to align all elements when printed. A value of 0
            disables this analysis and each elements will be printed in an
            unaligned format.
        float_format (str): default 'auto'. Determines the floating point
            format to use. Options: 'auto', 'fixed', 'scientific' and 'none'.
            'auto' mode determines the appropriate format based on the data.
            'fixed' uses fixed point format e.g. `-100.00`.
            'scientific' uses scientific notation e.g. `-1.123e+10`.
            'none' does not take care to display numbers in the same format.
            If `digits==0` this option is disregarded and the `float_format`
            is set to 'none'
        separator (str): default ','. Character used to delininate values.
        open_bracket (str): default '['. character used to open a
            tensor.
        close_bracket (str): default ']'. Character used to close a
            tensor.

    Raises:
        ValueError: if separator, open_bracket or close_bracket are
            not a single character.
        KeyError: if float_format is not one of the amiable options
            (see parameter docstring above)

    Returns:
        Tensor: The input tensor, unchanged.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    for name, char in (
        ("separator", separator),
        ("open_bracket", open_bracket),
        ("close_bracket", close_bracket),
    ):
        if len(char) != 1:
            raise ValueError(
                f"Parameter {name} must be 1 character exactly. "
                f"Not length {len(char)} and value {char}."
            )

    settings = ctx._get_op_settings("print_tensor")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "PrintTensor", 1, _ir.NumInputs(1, 1), 1
    )
    if title is None:
        title = f"print_{t.name}"

    float_format = FLOAT_FORMAT_MAP[float_format]
    fmt = PrintTensorFmt(
        summarise_threshold,
        edge_items,
        max_line_width,
        digits,
        float_format,
        separator,
        open_bracket,
        close_bracket,
    )

    op = pb_g.createConnectedOp_PrintTensorOp(
        {
            0: t.id,
        },
        {
            0: g._create_tensor_id("print_out"),
        },
        opid,
        print_self,
        print_gradient,
        title,
        fmt,
        settings,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
