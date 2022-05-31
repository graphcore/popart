# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional, Tuple
from typing_extensions import Literal

import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set

InterpolateType = Literal['nearest', 'linear', 'cubic']
InterpolateNearestType = Literal['round_prefer_floor', 'round_prefer_ceil',
                                 'floor', 'ceil']
InterpolateCoordinateTransformationType = Literal[
    'half_pixel', 'pytorch_half_pixel', 'align_corners', 'asymmetric',
    'tf_crop_and_resize']


def _convert_interpolate_type(mode: InterpolateType) -> _ir.op.ResizeMode:
    """Convert the given interpolate type to ir.op.ResizeMode.

    Args:
        mode (InterpolateType): The interpolate type to use

    Raises:
        ValueError: If an unsupported interpolation type is given

    Returns:
        _ir.op.ResizeMode: An internal IR object representing the interpolate type settings.
    """
    if mode == 'nearest':
        return _ir.op.ResizeMode.Nearest
    elif mode == 'linear':
        return _ir.op.ResizeMode.Linear
    elif mode == 'cubic':
        return _ir.op.ResizeMode.Cubic
    else:
        raise ValueError(
            f"interpolate type {mode} is not valid, must be nearest, linear or cubic."
        )


def _convert_interpolate_nearest_type(
        mode: InterpolateNearestType) -> _ir.op.ResizeNearestMode:
    """Convert the given interpolate nearest type to ir.op.ResizeNearestMode.

    Args:
        mode (InterpolateNearestType): The interpolate nearest type to use

    Raises:
        ValueError: If an unsupported interpolation type is given

    Returns:
        _ir.op.ResizeNearestMode: An internal IR object representing the interpolate nearest
            type settings.
    """
    if mode == 'round_prefer_floor':
        return _ir.op.ResizeNearestMode.RoundPreferFloor
    elif mode == 'round_prefer_ceil':
        return _ir.op.ResizeNearestMode.RoundPreferCeil
    elif mode == 'floor':
        return _ir.op.ResizeNearestMode.Floor
    elif mode == 'ceil':
        return _ir.op.ResizeNearestMode.Ceil
    else:
        raise ValueError(
            f"interpolate type {mode} is not valid, must be round_prefer_floor, round_prefer_ceil, floor or ceil."
        )


def _convert_interpolate_coordinate_transformation_type(
        mode: InterpolateCoordinateTransformationType
) -> _ir.op.ResizeCoordinateTransformationMode:
    """
    Convert the given interpolate coordinate transformation type.

    The type will be coverted to ir.op.ResizeCoordinateTransformationMode.

    Args:
        mode (InterpolateCoordinateTransformationType): The interpolate coordinate transformation
            type to use

    Raises:
        ValueError: If an unsupported interpolation type is given

    Returns:
        _ir.op.ResizeCoordinateTransformationMode: An internal IR object representing the
            interpolate coordinate transformation type settings.
    """
    if mode == 'half_pixel':
        return _ir.op.ResizeCoordinateTransformationMode.HalfPixel
    elif mode == 'pytorch_half_pixel':
        return _ir.op.ResizeCoordinateTransformationMode.PytorchHalfPixel
    elif mode == 'align_corners':
        return _ir.op.ResizeCoordinateTransformationMode.AlignCorners
    elif mode == 'asymmetric':
        return _ir.op.ResizeCoordinateTransformationMode.Asymmetric
    elif mode == 'tf_crop_and_resize':
        return _ir.op.ResizeCoordinateTransformationMode.TfCropAndResize
    else:
        raise ValueError(
            f"interpolate type {mode} is not valid, must be half_pixel, pytorch_half_pixel, align_corners, asymmetric or tf_crop_and_resize."
        )


@op_debug_context
def interpolate(
        t: Tensor,
        scale_factor: Optional[Tuple[float]] = (1.0, 1.0, 1.0, 1.0),
        mode: Optional[InterpolateType] = 'nearest',
        nearest_mode: Optional[InterpolateNearestType] = 'round_prefer_floor',
        coordinate_transformation_mode: Optional[
            InterpolateCoordinateTransformationType] = 'half_pixel') -> Tensor:
    """
    Interpolate the input tensor. Each dimension value of the output tensor is: output_dimension = floor(input_dimension * scale_factor).

    Args:
        t (Tensor):
            Input data tensor from previous layer.
        scale_factor (Tuple[float]):
            The scale array along each dimension. It takes value greater than or equal to 1. The number of elements of 'scales' should
            be the same as the rank of input 't'.
        mode (InterpolateType):
            The interpolate algorithm, three interpolation modes: nearest (default), linear and cubic.
        nearest_mode (InterpolateNearestType):
            Four modes: round_prefer_floor (default, as known as round half down), round_prefer_ceil (as known as round half up), floor, ceil.
            Only used by nearest interpolation. It indicates how to get "nearest" pixel in input tensor from x_original, so this attribute is
            valid only if "mode" is "nearest".
        coordinate_transformation_mode (InterpolateCoordinateTransformationType):
            This attribute describes how to transform the coordinate in the interpolated tensor to the coordinate in the original tensor.
            The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.

            Some variables are defined as follows.
            x_interpolated: the coordinate of axis x in the interpolated tensor.
            x_original: the coordinate of axis x in the original tensor.
            length_original: the length of the original tensor in axis x.
            length_interpolated: the length of the interpolated tensor in axis x.
            roi_x: roi_x = (start_x, end_x) of the axis x in input "roi".
            scale: scale = length_interpolated / length_original.

            if coordinate_transformation_mode is "half_pixel",
            x_original = (x_interpolated + 0.5) / scale - 0.5,

            if coordinate_transformation_mode is "pytorch_half_pixel",
            x_original = length_interpolated > 1 ? (x_interpolated + 0.5) / scale - 0.5 : 0,

            if coordinate_transformation_mode is "align_corners",
            x_original = x_interpolated * (length_original - 1) / (length_interpolated - 1),

            if coordinate_transformation_mode is "asymmetric",
            x_original = x_interpolated / scale,

            if coordinate_transformation_mode is "tf_crop_and_resize",
            x_original = length_interpolated > 1 ? start_x * (length_original - 1) + x_interpolated * (end_x - start_x) * (length_original - 1) / (length_interpolated - 1) :
                                                   0.5 * (start_x + end_x) * (length_original - 1).
    Returns:
        out (Tensor):
            Output data tensor after interpolate.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph
    check_in_graph(g, t=t)
    check_tensor_ipu_and_tile_set(t=t)

    settings = ctx._get_op_settings('interpolate')
    opid = _ir.OperatorIdentifier("ai.graphcore", "Resize", 1,
                                  _ir.NumInputs(1, 1), 1)
    sample_type = _convert_interpolate_type(mode)
    nearest_type = _convert_interpolate_nearest_type(nearest_mode)
    coordinate_transformation_type = _convert_interpolate_coordinate_transformation_type(
        coordinate_transformation_mode)
    op = pb_g.createConnectedOp_ResizeOp(
        {0: t.id}, {0: g._create_tensor_id("interpolate_out")}, opid, settings,
        sample_type, scale_factor, nearest_type,
        coordinate_transformation_type)

    return Tensor._from_pb_tensor(op.outTensor(0))
