# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
from .conv import PadType
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


def receptive_attributes(strides, pads, out_pads, dilations, in_dilations,
                         auto_pad, ceil_mode):
    receptive_attributes = _ir.op.ReceptiveOpAttributes()
    if strides:
        receptive_attributes.strides = strides
    if pads:
        receptive_attributes.pads = list(pads)
    if out_pads:
        receptive_attributes.outPads = list(out_pads)
    if dilations:
        receptive_attributes.dilations = list(dilations)
    if in_dilations:
        receptive_attributes.inDilations = list(in_dilations)
    if auto_pad:
        pad_dict = {
            'not_set': 'NOTSET',
            'same_upper': 'SAME_UPPER',
            'same_lower': 'SAME_LOWER',
            'valid': 'VALID'
        }
        receptive_attributes.auto_pad = pad_dict[auto_pad]
    if ceil_mode:
        receptive_attributes.ceil_mode = int(ceil_mode)
    return receptive_attributes


@op_debug_context
def average_pool(t: Tensor,
                 kernel_size: Tuple[int],
                 stride: Optional[Tuple[int]] = None,
                 padding: Optional[Tuple[int]] = None,
                 out_pads: Optional[Tuple[int]] = None,
                 dilation: Optional[Tuple[int]] = None,
                 in_dilations: Optional[Tuple[int]] = None,
                 auto_pad: Optional[PadType] = 'not_set',
                 ceil_mode: Optional[bool] = None) -> Tensor:
    """
    average_pool consumes an input tensor `t` and applies average pooling across the tensor according to kernel sizes,
    stride sizes, and pad lengths.
    Average pooling consisting of computing the average on all values of a subset of the input tensor according to
    the kernel size and downsampling the data into the output tensor Y for further processing.

    Args:
        t (Tensor):
            Input data tensor from previous layer; If the input is a 3D tensor, the size is (N, C, L), where N is the batch size,
            C is the number of channel, L is the length; If the input is a 2D image, the size is (N, C, H, W), where N is the batch size,
            C is the number of channel, H and W are the height and width; If the input is a 3D image, the size is (N, C, D, H, W), where
            N is the batch size, C is the number of channel, D is the depth, H and W are the height and width.
        kernel_size (Tuple[int]):
            The size of the kernel along each axis.
        stride (Tuple[int]):
            Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
        padding (Tuple[int]):
            Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
            The value represent the number of pixels added to the beginning and end part of the corresponding axis. `padding`
            format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added
            at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`.
        out_pads (Tuple[int]):
            The output padding for pooling.
        dilations (Tuple[int]):
            dilation value along each spatial axis of the filter.
        in_dilations (Tuple[int]):
            The input dilations attributes along each spatial axis of the filter.
        auto_pad (Literal):
            auto_pad must be either not_set, same_upper, same_lower or valid. Where default value is not_set, which means explicit
            padding is used. same_upper or same_lower mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])`
            for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even
            or odd). In case the padding is an odd number, the extra padding is added at the end for same_upper and at the beginning
            for same_lower.
        ceil_mode (bool):
            Whether to use ceil or floor (default) to compute the output shape.
    Returns:
        out (Tensor):
            Output data tensor from average pooling across the input tensor. Dimensions will vary based on various kernel, stride, and
            pad sizes. Floor value of the dimension is used.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    check_in_graph(g, t=t)
    check_tensor_ipu_and_tile_set(t=t)

    settings = ctx._get_op_settings('averagepool')
    opid = _ir.OperatorIdentifier("ai.onnx", "AveragePool", 11,
                                  _ir.NumInputs(1, 1), 1)
    recp_attr = receptive_attributes(stride, padding, out_pads, dilation,
                                     in_dilations, auto_pad, ceil_mode)
    op = pb_g.createConnectedOp_AveragePoolOp(
        {
            0: t.id,
        }, {
            0: g._create_tensor_id("averagepool_out"),
        }, opid, 0, list(kernel_size), recp_attr, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))


@op_debug_context
def max_pool(t: Tensor,
             kernel_size: Tuple[int],
             stride: Optional[Tuple[int]] = None,
             padding: Optional[Tuple[int]] = None,
             out_pads: Optional[Tuple[int]] = None,
             dilation: Optional[Tuple[int]] = None,
             in_dilations: Optional[Tuple[int]] = None,
             auto_pad: Optional[PadType] = 'not_set',
             ceil_mode: Optional[bool] = None,
             storage_order: Optional[bool] = None) -> Tensor:
    """
    max_pool consumes an input tensor `t` and applies max pooling across the tensor according to kernel sizes,
    stride sizes, and pad lengths. 
    Max pooling consisting of computing the max on all values of a subset of the input tensor according to
    the kernel size and downsampling the data into the output tensor Y for further processing.

    Args:
        t (Tensor):
            Input data tensor from previous layer; If the input is a 3D tensor, the size is (N, C, L), where N is the batch size,
            C is the number of channel, L is the length; If the input is a 2D image, the size is (N, C, H, W), where N is the batch size,
            C is the number of channel, H and W are the height and width; If the input is a 3D image, the size is (N, C, D, H, W), where
            N is the batch size, C is the number of channel, D is the depth, H and W are the height and width.
        kernel_size (Tuple[int]):
            The size of the kernel along each axis.
        stride (Tuple[int]):
            Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
        padding (Tuple[int]):
            Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value
            represent the number of pixels added to the beginning and end part of the corresponding axis. `padding` format should be as
            follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i`
            and xi_end, the number of pixels added at the end of axis `i`.
        out_pads (Tuple[int]):
            The output padding for pooling.
        dilation (Tuple[int]):
            dilation value along each spatial axis of the filter.
        in_dilations (Tuple[int]):
            The input dilations attributes along each spatial axis of the filter.
        auto_pad (Literal):
            auto_pad must be either not_set, same_upper, same_lower or valid. Where default value is not_set, which means explicit
            padding is used. same_upper or same_lower mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])`
            for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even
            or odd). In case the padding is an odd number, the extra padding is added at the end for same_upper and at the beginning
            for same_lower.
        ceil_mode (bool):
            Whether to use ceil or floor (default) to compute the output shape.
        storage_order (bool):
            The storage order of the tensor. 0 is row major, and 1 is column major, default is false.
    Returns:
        out (Tensor):
            Output data tensor from max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes.
            Floor value of the dimension is used.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    check_in_graph(g, t=t)
    check_tensor_ipu_and_tile_set(t=t)

    settings = ctx._get_op_settings('maxpool')
    opid = _ir.OperatorIdentifier("ai.onnx", "MaxPool", 11, _ir.NumInputs(
        1, 1), 1)
    recp_attr = receptive_attributes(stride, padding, out_pads, dilation,
                                     in_dilations, auto_pad, ceil_mode)
    storage_order_int = 0
    if storage_order:
        storage_order_int = 1
    op = pb_g.createConnectedOp_MaxPoolOp(
        {
            0: t.id,
        }, {
            0: g._create_tensor_id("maxpool_out"),
        }, opid, list(kernel_size), storage_order_int, recp_attr, settings)
    return Tensor._from_pb_tensor(op.outTensor(0))
