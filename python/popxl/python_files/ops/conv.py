# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, List
from typing_extensions import Literal

import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set

PadType = Literal["not_set", "same_upper", "same_lower", "valid"]


def multi_conv_options(
    sess_opts,
    attr_param,
    available_memory_proportions,
    partials_types,
    enable_conv_dithering,
):
    multi_conv_options = _ir.op.MultiConvOptions(sess_opts, attr_param)
    if available_memory_proportions:
        multi_conv_options.availableMemoryProportions = available_memory_proportions
    if partials_types:
        multi_conv_options.partialsTypes = partials_types
    if enable_conv_dithering:
        multi_conv_options.enableConvDithering = enable_conv_dithering
    return multi_conv_options


def _convert_pad_type(mode: PadType) -> _ir.op.AutoPad:
    """
    Convert the given pad type to ir.op.AutoPad.

    Args:
        mode (PadType): The pad type to use

    Raises:
        ValueError: If an unsupported pad type is given

    Returns:
        _ir.op.AutoPad: An internal IR object representing the pad type settings.
    """
    if mode == "not_set":
        return _ir.op.AutoPad.NOTSET
    elif mode == "same_upper":
        return _ir.op.AutoPad.SAME_UPPER
    elif mode == "same_lower":
        return _ir.op.AutoPad.SAME_LOWER
    elif mode == "valid":
        return _ir.op.AutoPad.VALID
    else:
        raise ValueError(
            f"pad type {mode} is not valid, must be not_set, same_upper, same_lower or valid"
        )


@op_debug_context
def conv(
    t: Tensor,
    weight: Tensor,
    stride: Optional[Tuple[int]] = (1, 1),
    padding: Optional[Tuple[int]] = (0, 0, 0, 0),
    dilation: Optional[Tuple[int]] = (1, 1),
    groups: Optional[int] = 1,
    pad_type: Optional[PadType] = "not_set",
    available_memory_proportions: Optional[List[float]] = None,
    partials_types: Optional[List[str]] = None,
    enable_conv_dithering: Optional[List[int]] = None,
) -> Tensor:
    """
    Use the convolution operator on a tensor.

    The convolution operator consumes an input tensor and a filter, and computes the output.

    Args:
        t (Tensor):
            Input data tensor from previous layer;
            If the input is a 3D tensor, the size is (N, C, L), where N is the batch size, C is the
            number of channel, L is the length;
            If the input is a 2D image, the size is (N, C, H, W), where N is the batch size, C is
            the number of channel, H and W are the height and width;
            If the input is a 3D image, the size is (N, C, D, H, W), where N is the batch size,
            C is the number of channel, D is the depth, H and W are the height and width.
        weight (Tensor):
            The weight tensor that will be used in the convolutions;
            If the input is a 3D tensor, the weight size is (M, C/group, k), where C is the number
            of channels, k is the length of the kernel, M is the number of feature maps.
            If the input is a 2D image, the weight size is (M, C/group, kH, kW), where C is the
            number of channels, kH and kW are the height and width of the kernel, M is the number
            of feature maps.
            If the input is a 3D image, the weight size is (M, C/group, kD, kH, kW), where C is the
            number of channels, kD, kH and kW are the depth, height and width of the kernel, M is
            the number of feature maps.
        stride (Tuple[int]):
            Stride along each spatial axis.
        padding (Tuple[int]):
            Padding for the beginning and ending along each spatial axis, it can take any value
            greater than or equal to 0.
            The value represent the number of pixels added to the beginning and end part of the
            corresponding axis.
            `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where
            xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number
            of pixels added at the end of axis `i`.
        dilation (Tuple[int]):
            dilation value along each spatial axis of the filter.
        groups (int(default is 1)):
            number of groups input channels and output channels are divided into.
        pad_type (PadType(default is not_set)):
            pad_type must be either "not_set", "same_upper", "same_lower" or "valid".
            The default value is "not_set", which means explicit padding is used.
            "same_upper" or "same_lower" mean pad the input so that
            `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
            The padding is split between the two sides equally or almost equally
            (depending on whether it is even or odd).
            In the case that the padding is an odd number, the extra padding is added at the end for
            "same_upper" and at the beginning for "same_lower".
        available_memory_proportions (List[float]):
            The available memory proportions per conv, each [0, 1).
        partials_types (List[str]):
            The partials type per convolution, choose between half and float.
        enable_conv_dithering (List[int]):
            Enable convolution dithering per convolution.
            If true, then convolutions with different parameters will be laid out from different tiles
            in an effort to improve tile balance in models.
    Returns:
        Tensor:
            A tensor that contains the result of the convolution. The output dimensions are functions of the kernel size,
            stride size, and pad lengths.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    check_in_graph(g, t=t, weight=weight)
    check_tensor_ipu_and_tile_set(t=t, weight=weight)

    settings = ctx._get_op_settings("conv")
    opid = _ir.OperatorIdentifier("ai.onnx", "Conv", 11, _ir.NumInputs(1, 1), 1)
    auto_pad = _convert_pad_type(pad_type)
    sess_opts = g.ir._pb_ir.getSessionOptions().convolutionOptions
    attr_param = _ir.op.Attributes()
    options = multi_conv_options(
        sess_opts,
        attr_param,
        available_memory_proportions,
        partials_types,
        enable_conv_dithering,
    )
    op = pb_g.createConnectedOp_ConvOp(
        {0: t.id, 1: weight.id},
        {
            0: g._create_tensor_id("conv_out"),
        },
        opid,
        settings,
        list(stride),
        list(padding),
        list(dilation),
        groups,
        auto_pad,
        options,
    )

    return Tensor._from_pb_tensor(op.outTensor(0))
