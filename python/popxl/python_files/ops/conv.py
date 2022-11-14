# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple, List
from typing_extensions import Literal

import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from popxl.dtypes import float8_143, float8_152, int32
from .utils import check_in_graph, check_tensor_ipu_and_tile_set

PadType = Literal["not_set", "same_upper", "same_lower", "valid"]


def _check_pow2scaled_input_properties(t: Tensor, weight: Tensor, log2_scale: Tensor):
    invalid_dtype = (
        t.dtype
        not in {
            float8_143,
            float8_152,
        }
    ) or (weight.dtype not in {float8_143, float8_152})
    if invalid_dtype:
        raise TypeError(
            "Scaled conv operands must be of dtype popxl.float8_143 or popxl.float8_152"
        )
    elif log2_scale.rank > 0:
        raise ValueError("Log2 scale argument must be a scalar tensor")
    elif log2_scale.dtype != int32:
        raise TypeError("Log2 scale tensor must be of type popxl.int32")


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


@op_debug_context
def conv_pow2scaled(
    t: Tensor,
    weight: Tensor,
    log2_scale: Tensor,
    stride: Optional[Tuple[int]] = (1, 1),
    padding: Optional[Tuple[int]] = (0, 0, 0, 0),
    dilation: Optional[Tuple[int]] = (1, 1),
    groups: Optional[int] = 1,
    pad_type: Optional[PadType] = "not_set",
    available_memory_proportions: Optional[List[float]] = None,
    enable_conv_dithering: Optional[List[int]] = None,
) -> Tensor:
    """
    Perform a scaled convolution on a float8 tensor.

    The convolution operator consumes an input tensor, a filter and
    computes the output. The dtype of the input tensor and filter
    must be one of `popxl.float8_143` or `popxl.float8_152`.

    The result of the convolution is scaled by `pow2(log2_scale)`
    before it is converted to float16.

    The `log2_scale` must be a scalar tensor of type `popxl.int32` and contain a runtime value in
    the range `[-32, 32)`

    Args:
        t (Tensor):
            Input data tensor from previous layer of type either `popxl.float8_143` or `popxl.float8_152`;
            If the input is a 3D tensor, the size is (N, C, L), where N is the batch size, C is the
            number of channel, L is the length;
            If the input is a 2D image, the size is (N, C, H, W), where N is the batch size, C is
            the number of channel, H and W are the height and width;
            If the input is a 3D image, the size is (N, C, D, H, W), where N is the batch size,
            C is the number of channel, D is the depth, H and W are the height and width.
        weight (Tensor):
            The weight tensor that will be used in the convolutions of type either `popxl.float8_143` or `popxl.float8_152`;
            If the input is a 3D tensor, the weight size is (M, C/group, k), where C is the number
            of channels, k is the length of the kernel, M is the number of feature maps.
            If the input is a 2D image, the weight size is (M, C/group, kH, kW), where C is the
            number of channels, kH and kW are the height and width of the kernel, M is the number
            of feature maps.
            If the input is a 3D image, the weight size is (M, C/group, kD, kH, kW), where C is the
            number of channels, kD, kH and kW are the depth, height and width of the kernel, M is
            the number of feature maps.
        log2_scale (Tensor):
            32-bit integer power-of-two exponent, where the convolution
            output is multiplied by `pow2(log2_scale)` before conversion to float16.
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
        enable_conv_dithering (List[int]):
            Enable convolution dithering per convolution.
            If true, then convolutions with different parameters will be laid out from different tiles
            in an effort to improve tile balance in models.
    Returns:
        Tensor:
            A tensor that contains the result of the convolution of type `popxl.float16`. The output
            dimensions are functions of the kernel size, stride size, and pad lengths.

    Raises:
        TypeError: If the tensor or weight tensors do not have a dtype in
            `{popxl.float8_143, popxl.float8_152}`, or if the `log2_scale` tensor
            does not have dtype `popxl.int32`
        ValueError: If `log2_scale` is not a scalar tensor.
    """
    _check_pow2scaled_input_properties(t, weight, log2_scale)

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    check_in_graph(g, t=t, weight=weight, log2_scale=log2_scale)
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
        partials_types=["half"],
        enable_conv_dithering=enable_conv_dithering,
    )
    op = pb_g.createConnectedOp_ConvOp(
        {0: t.id, 1: weight.id, 2: log2_scale.id},
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


def conv_transpose(
    t: Tensor,
    weight: Tensor,
    stride: Optional[Tuple[int]] = (1, 1),
    padding: Optional[Tuple[int]] = (0, 0, 0, 0),
    dilation: Optional[Tuple[int]] = (1, 1),
    groups: Optional[int] = 1,
    pad_type: Optional[PadType] = "not_set",
    output_padding: Optional[Tuple[int]] = (),
    output_shape: Optional[Tuple[int]] = (),
    available_memory_proportions: Optional[List[float]] = None,
    partials_types: Optional[List[str]] = None,
    enable_conv_dithering: Optional[List[int]] = None,
):
    """Perform a convolution transpose operation on a tensor.

    The convolution transpose operator consumes an input tensor and a filter, and computes the output.

    If the `padding` parameter is provided the shape of the output is auto generated. `output_shape`
    can also be explicitly specified in which case `padding` values are auto generated. See attribute
    descriptions for more details.

    See also `PyTorch Tensor.ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#convtranspose2d>`__,
    `ONNX ConvTranspose <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose>`__.

    Attributes:
        t (Tensor):
            Input data tensor from a previous layer.
            If the input is a 3D tensor, the size is (N, C, L), where N is the batch size, C is the
            number of channels, L is the length;
            If the input is a 2D image, the size is (N, C, H, W), where N is the batch size, C is
            the number of channels, H and W are the height and width;
            If the input is a 3D image, the size is (N, C, D, H, W), where N is the batch size,
            C is the number of channels, D is the depth, H and W are the height and width.
        weight (Tensor):
            The weight tensor that will be used in the convolutions.
            If the input is a 3D tensor, the weight size is (M, C/group, k), where C is the number
            of channels, k is the length of the kernel, M is the number of feature maps.
            If the input is a 2D image, the weight size is (M, C/group, kH, kW), where C is the
            number of channels, kH and kW are the height and width of the kernel, M is the number
            of feature maps.
            If the input is a 3D image, the weight size is (M, C/group, kD, kH, kW), where C is the
            number of channels, kD, kH and kW are the depth, height and width of the kernel, M is
            the number of feature maps.
        padding (Tuple[int]):
            Padding for the beginning and ending along each spatial axis, it can take any value
            greater than or equal to 0.
            The value represent the number of pixels added to the beginning and end part of the
            corresponding axis.
            `pads` format should be `[x1_begin, x2_begin...x1_end, x2_end,...]`, where
            `xi_begin` is the number of pixels added at the beginning of axis `i` and `xi_end` is
            the number of pixels added at the end of axis `i`.
            If the pads parameter is provided the shape of the output is auto generated. `See ONNX Conv Transpose
            <https://github.com/onnx/onnx/blob/main/docs/Operators.md#convtranspose>`__ for details.
        dilation (Tuple[int]):
            Dilation value along each spatial axis of the filter.
        groups (int(default is 1)):
            Number of groups input channels and output channels are divided into.
        pad_type (PadType(default is not_set)):
            The `pad_type` must be either "not_set", "same_upper", "same_lower" or "valid".
            The default value is "not_set", which means explicit padding is used.
            "same_upper" or "same_lower" mean pad the input such that
            `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
            The padding is split between the two sides equally or almost equally
            (depending on whether it is even or odd).
            In the case that the padding is an odd number, the extra padding is added at the end for
            "same_upper" and at the beginning for "same_lower".
        output_padding (Tuple[int]):
            Additional elements added to the side with higher coordinate indices in the output. Each padding
            value in `output_padding` must be strictly less than the corresponding stride/dilation dimension.
            Note that this attribute doesn't directly affect the computed output values. It only controls the
            selection of the computed values, so changing this attribute only adds or removes output elements.
            If `output_shape` is explicitly provided, `output_padding` does not contribute additional size to
            `output_shape` but participates in the computation of the needed padding amount.
        output_shape (Tuple[int]):
            The shape of the output can be explicitly set which will cause padding values to be auto generated.
            If output_shape is specified pads values are ignored. `See ONNX Conv Transpose
            <https://github.com/onnx/onnx/blob/main/docs/Operators.md#convtranspose>`__ for details on how
            padding is generated.
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
            Output data tensor that contains the result of the convolution. The output dimensions are functions
             of the kernel size, stride size, pad lengths and group count.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    check_in_graph(g, t=t, weight=weight)
    check_tensor_ipu_and_tile_set(t=t, weight=weight)

    settings = ctx._get_op_settings("conv")
    opid = _ir.OperatorIdentifier(
        "ai.onnx", "ConvTranspose", 11, _ir.NumInputs(1, 1), 1
    )
    auto_pad = _convert_pad_type(pad_type)
    sess_opts = g.ir._pb_ir.getSessionOptions().convolutionOptions

    attr_param = _ir.op.Attributes()
    options = multi_conv_options(
        sess_opts,
        attr_param,
        available_memory_proportions,
        partials_types=partials_types,
        enable_conv_dithering=enable_conv_dithering,
    )

    op = pb_g.createConnectedOp_ConvTransposeOp(
        {0: t.id, 1: weight.id},
        {0: g._create_tensor_id("convtranspose_out")},
        opid,
        settings,
        list(stride),
        list(padding),
        list(dilation),
        groups,
        auto_pad,
        list(output_padding),
        list(output_shape),
        options,
    )
    return Tensor._from_pb_tensor(op.outTensor(0))


def conv_transpose_pow2scaled(
    t: Tensor,
    weight: Tensor,
    log2_scale: Tensor,
    stride: Optional[Tuple[int]] = (1, 1),
    padding: Optional[Tuple[int]] = (0, 0, 0, 0),
    dilation: Optional[Tuple[int]] = (1, 1),
    groups: Optional[int] = 1,
    pad_type: Optional[PadType] = "not_set",
    output_padding: Optional[Tuple[int]] = (),
    output_shape: Optional[Tuple[int]] = (),
    available_memory_proportions: Optional[List[float]] = None,
    enable_conv_dithering: Optional[List[int]] = None,
):
    """Perform a single transposed and scaled convolution operation on a tensor.

    This operator consumes an input, weight, and log2 scale tensor to compute a transposed
    convolution, then scales the convolution output by `pow2(log2_scale)` before converting to float16.

    The dtype of the input `t` and weight tensor must be one of `popxl.float8_143` or `popxl.float8_152`.
    The `log2_scale` must be a scalar tensor of type `popxl.int32` and contain a runtime value in
    the range `[-32, 32)`

    If the `padding` parameter is provided the shape of the output is auto generated. `output_shape`
    can also be explicitly specified in which case `padding` values are auto generated. See attribute
    descriptions for more details.

    See also `PyTorch Tensor.ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#convtranspose2d>`__,
    `ONNX ConvTranspose <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose>`__.

    Attributes:
        t (Tensor):
            Input data tensor from previous layer of type either `popxl.float8_143` or `popxl.float8_152`;
            If the input is a 3D tensor, the size is (N, C, L), where N is the batch size, C is the
            number of channels, L is the length;
            If the input is a 2D image, the size is (N, C, H, W), where N is the batch size, C is
            the number of channels, H and W are the height and width;
            If the input is a 3D image, the size is (N, C, D, H, W), where N is the batch size,
            C is the number of channels, D is the depth, H and W are the height and width.
        weight (Tensor):
            The weight tensor that will be used as a kernel in the convolution, of dtype either
            `popxl.float8_143` or `popxl.float8_152`;
            If the input is a 3D tensor, the weight size is (M, C/group, k), where C is the number
            of channels, k is the length of the kernel, M is the number of feature maps.
            If the input is a 2D image, the weight size is (M, C/group, kH, kW), where C is the
            number of channels, kH and kW are the height and width of the kernel, M is the number
            of feature maps.
            If the input is a 3D image, the weight size is (M, C/group, kD, kH, kW), where C is the
            number of channels, kD, kH and kW are the depth, height and width of the kernel, M is
            the number of feature maps.
        log2_scale (Tensor):
            32-bit integer power-of-two exponent, where the convolution
            output is multiplied by `pow2(log2_scale)` before conversion to float16. Must be of dtype
            `popxl.int32`.
        padding (Tuple[int]):
            Padding for the beginning and ending along each spatial axis, it can take any value
            greater than or equal to 0.
            The value represent the number of pixels added to the beginning and end part of the
            corresponding axis.
            `pads` format should be `[x1_begin, x2_begin...x1_end, x2_end,...]`, where
            `xi_begin` is the number of pixels added at the beginning of axis `i` and `xi_end` is
            the number of pixels added at the end of axis `i`.
            If the pads parameter is provided the shape of the output is auto generated. `See ONNX Conv Transpose
            <https://github.com/onnx/onnx/blob/main/docs/Operators.md#convtranspose>`__ for details.
        dilation (Tuple[int]):
            Dilation value along each spatial axis of the filter.
        groups (int(default is 1)):
            Number of groups input channels and output channels are divided into.
        pad_type (PadType(default is not_set)):
            The `pad_type` must be either "not_set", "same_upper", "same_lower" or "valid".
            The default value is "not_set", which means explicit padding is used.
            "same_upper" or "same_lower" mean pad the input such that
            `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
            The padding is split between the two sides equally or almost equally
            (depending on whether it is even or odd).
            In the case that the padding is an odd number, the extra padding is added at the end for
            "same_upper" and at the beginning for "same_lower".
        output_padding (Tuple[int]):
            Additional elements added to the side with higher coordinate indices in the output. Each padding
            value in `output_padding` must be strictly less than the corresponding stride/dilation dimension.
            Note that this attribute doesn't directly affect the computed output values. It only controls the
            selection of the computed values, so changing this attribute only adds or removes output elements.
            If `output_shape` is explicitly provided, `output_padding` does not contribute additional size to
            `output_shape` but participates in the computation of the needed padding amount.
        output_shape (Tuple[int]):
            The shape of the output can be explicitly set which will cause padding values to be auto generated.
            If output_shape is specified pads values are ignored. `See ONNX Conv Transpose
            <https://github.com/onnx/onnx/blob/main/docs/Operators.md#convtranspose>`__ for details on how
            padding is generated.
        available_memory_proportions (List[float]):
            The available memory proportions per conv, each [0, 1).
        enable_conv_dithering (List[int]):
            Enable convolution dithering per convolution.
            If true, then convolutions with different parameters will be laid out from different tiles
            in an effort to improve tile balance in models.

    Returns:
        Tensor:
            Output data tensor that contains the result of the convolution. The output dimensions are functions
             of the kernel size, stride size, pad lengths and group count.
    Raises:
        TypeError: If the tensor or weight tensors do not have a dtype in
            `{popxl.float8_143, popxl.float8_152}`, or if the `log2_scale` tensor
            does not have dtype `popxl.int32`
        ValueError: If `log2_scale` is not a scalar tensor.
    """
    _check_pow2scaled_input_properties(t, weight, log2_scale)

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    check_in_graph(g, t=t, weight=weight)
    check_tensor_ipu_and_tile_set(t=t, weight=weight)

    settings = ctx._get_op_settings("conv")
    opid = _ir.OperatorIdentifier(
        "ai.onnx", "ConvTranspose", 11, _ir.NumInputs(1, 1), 1
    )
    auto_pad = _convert_pad_type(pad_type)
    sess_opts = g.ir._pb_ir.getSessionOptions().convolutionOptions

    attr_param = _ir.op.Attributes()
    options = multi_conv_options(
        sess_opts,
        attr_param,
        available_memory_proportions,
        partials_types=["half"],
        enable_conv_dithering=enable_conv_dithering,
    )

    op = pb_g.createConnectedOp_ConvTransposeOp(
        {0: t.id, 1: weight.id, 2: log2_scale.id},
        {0: g._create_tensor_id("convtranspose_out")},
        opid,
        settings,
        list(stride),
        list(padding),
        list(dilation),
        groups,
        auto_pad,
        list(output_padding),
        list(output_shape),
        options,
    )
    return Tensor._from_pb_tensor(op.outTensor(0))
