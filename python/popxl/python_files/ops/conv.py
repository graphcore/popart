# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
from typing_extensions import Literal

import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set

PadType = Literal['NOTSET', 'SAME_UPPER', 'SAME_LOWER', 'VALID']


@op_debug_context
def conv2d(t: Tensor,
           weight: Tensor,
           stride: Optional[Tuple[int]] = (1, 1),
           padding: Optional[Tuple[int]] = (0, 0, 0, 0),
           dilation: Optional[Tuple[int]] = (1, 1),
           groups: Optional[int] = None,
           pad_type: Optional[PadType] = None) -> Tensor:
    """
    The convolution operator consumes an input tensor and a filter, and computes the output.

    Args:
        t (Tensor):
           Input data tensor from previous layer; has size (N, C, H, W), where N is the batch size, C is the number of channels,
           and H and W are the height and width. Note that this is for the 2D image.
        weight (Tensor):
            The weight tensor that will be used in the convolutions; has size (M, C/group, kH, kW), where C is the number of channels,
            and kH and kW are the height and width of the kernel, and M is the number of feature maps.
        stride (Tuple[int]):
            Stride along each spatial axis.
        padding (Tuple[int]):
            Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value
            represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as
            follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i`
            and xi_end, the number of pixels added at the end of axis `i`.
        dilation (Tuple[int]):
            dilation value along each spatial axis of the filter.
        groups (int(default is 1)):
            number of groups input channels and output channels are divided into.
        pad_type (PadType(default is NOTSET)):
            pad_type must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit
            padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])`
            for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even or
            odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for
            SAME_LOWER.
    Returns:
        out (Tensor):
            Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, 
            stride size, and pad lengths.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    check_in_graph(g, t=t, weight=weight)
    check_tensor_ipu_and_tile_set(t=t, weight=weight)

    settings = ctx._get_op_settings('conv2d')
    opid = _ir.OperatorIdentifier("ai.onnx", "Conv", 11, _ir.NumInputs(1, 1),
                                  1)
    auto_pad = _convert_pad_type(pad_type)
    sess_opts = g.ir._pb_ir.getSessionOptions().convolutionOptions
    attr_param = _ir.op.Attributes()
    multi_conv_options = _ir.op.MultiConvOptions(sess_opts, attr_param)
    op = pb_g.createConnectedOp_ConvOp({
        0: t.id,
        1: weight.id
    }, {
        0: g._create_tensor_id("conv_out"),
    }, opid, settings, list(stride), list(padding), list(dilation), groups,
                                       auto_pad, multi_conv_options)

    return Tensor._from_pb_tensor(op.outTensor(0))


def _convert_pad_type(mode: PadType) -> _ir.op.AutoPad:
    """Convert the given pad type to ir.op.AutoPad

    Args:
        mode (PadType): The pad type to use

    Returns:
        _ir.op.AutoPad: An internal IR object representing the pad type settings.
    """
    if mode == 'NOTSET':
        return _ir.op.AutoPad.NOTSET
    elif mode == 'SAME_UPPER':
        return _ir.op.AutoPad.SAME_UPPER
    elif mode == 'SAME_LOWER':
        return _ir.op.AutoPad.SAME_LOWER
    elif mode == 'VALID':
        return _ir.op.AutoPad.VALID
    else:
        raise ValueError(
            f"pad type {mode} is not valid, must be NOTSET, SAME_UPPER, SAME_LOWER or VALID"
        )
