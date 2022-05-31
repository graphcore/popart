# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Tuple

import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def roi_align(t: Tensor, rois: Tensor, batch_index: Tensor,
              output_size: Tuple[int], spatial_scale: float,
              sampling_ratio: int) -> Tensor:
    """
    Apply pooling across each region of interest (RoIs).

    roi_align consumes an input tensor `t` and region of interests to apply pooling across each
    RoI.
    Currently only supports average pooling, max pooling is not supported yet.

    Args:
        t (Tensor):
            Input data tensor from the previous operator;
            4-D feature map of shape (N, C, H, W), where N is the batch size, C is the number of
            channels, and H and W are the height and the width of the data.
        rois (Tensor):
            RoIs (Regions of Interest) to pool over;
            rois is 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], ...].
            The RoIs' coordinates are in the coordinate system of the input image.
            Each coordinate set has a 1:1 correspondence with 'batch_index' input.
        batch_index (Tensor):
            1-D tensor of shape (numRois,) with each element denoting the index of the
            corresponding image in the batch.
        output_size (Tuple[int]):
            Pooled output height and width.
        spatial_scale (float):
            Multiplicative spatial scale factor to translate ROI coordinates from their input
            spatial scale to the scale used when pooling, i.e., spatial scale of the input feature
            map `t` relative to the input image.
        sampling_ratio (int):
            Number of sampling points in the interpolation grid used to compute the output value of
            each pooled output bin.
    Returns:
        out (Tensor):
            RoI pooled output Y, 4-D tensor of shape
            (numRois, channels, aligned_height_, aligned_width_).
            The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI
            t[r-1].
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, rois=rois, batch_index=batch_index)
    check_tensor_ipu_and_tile_set(t=t, rois=rois, batch_index=batch_index)

    settings = ctx._get_op_settings('roi_align')
    opid = _ir.OperatorIdentifier("ai.onnx", "RoiAlign", 10, _ir.NumInputs(
        2, 2), 1)
    aligned_height = output_size[0]
    aligned_width = output_size[1]
    op = pb_g.createConnectedOp_RoiAlignOp(
        {
            0: t.id,
            1: rois.id,
            2: batch_index.id
        }, {
            0: g._create_tensor_id("roi_align_out"),
        }, opid, settings, spatial_scale, sampling_ratio, aligned_height,
        aligned_width)

    return Tensor._from_pb_tensor(op.outTensor(0))
