// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <onnx/defs/schema.h>
#include <onnx/defs/shape_inference.h>
#include <string>
#include <vector>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/op/receptive.hpp>

#include "onnx/onnx_pb.h"
#include "popart/operatoridentifier.hpp"
#include "popart/util.hpp"

/*
 * This registers ONNX operators and shape inference for the Graphcore custom
 * operators.
 */

namespace ONNX_NAMESPACE {

/**
 * Helper for extracting INT attributes.
 *
 * \param ctx The inference context.
 * \param attrName The name of the attribute to be extracted.
 * \return The value of the attribute.
 */
auto getIntAttribute(InferenceContext &ctx, const std::string &attrName) {
  auto proto = ctx.getAttribute(attrName);

  if (proto && proto->has_i()) {
    return proto->i();
  }

  throw popart::internal_error("{} must be supplied as an integer", attrName);
}

void SubsampleShapeInference(InferenceContext &ctx);
void GroupNormalizationShapeInference(InferenceContext &ctx);
void PrintTensorShapeInference(InferenceContext &ctx);
void ScaleShapeInference(InferenceContext &ctx);
void LSTMShapeInference(InferenceContext &ctx);
void GeluShapeInference(InferenceContext &ctx);
void GeluErfShapeInference(InferenceContext &ctx);
void DetachShapeInference(InferenceContext &ctx);
void CallShapeInference(InferenceContext &ctx);
void DynamicUpdateShapeInference(InferenceContext &ctx);
void DynamicSliceShapeInference(InferenceContext &ctx);
void DynamicZeroShapeInference(InferenceContext &ctx);
void DynamicAddShapeInference(InferenceContext &ctx);
void SequenceSliceInference(InferenceContext &ctx);
void MultiConvShapeInference(InferenceContext &ctx);
void NopShapeInference(InferenceContext &ctx);
void ShapedDropoutShapeInference(InferenceContext &ctx);
void Atan2ShapeInference(InferenceContext &ctx);
void DepthToSpaceShapeInference(InferenceContext &ctx);
void Expm1ShapeInference(InferenceContext &ctx);
void Log1pShapeInference(InferenceContext &ctx);
void ReshapeShapeInference(InferenceContext &ctx);
void ReverseShapeInference(InferenceContext &ctx);
void SliceShapeInference(InferenceContext &ctx);
void ScatterReduceShapeInference(InferenceContext &ctx);
void RemainderShapeInference(InferenceContext &ctx);
void FmodShapeInference(InferenceContext &ctx);
void BitwiseNotShapeInference(InferenceContext &ctx);
void RoundShapeInference(InferenceContext &ctx);
void CtcBeamSearchDecoderShapeInference(InferenceContext &ctx);
void CtcLossShapeInference(InferenceContext &ctx);
void ReduceMedianShapeInference(InferenceContext &ctx);
void CopyVarUpdateShapeInference(InferenceContext &ctx);
void SwishShapeInference(InferenceContext &ctx);
void BucketizeShapeInference(InferenceContext &ctx);
void SortShapeInference(InferenceContext &ctx);

void SubsampleShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  // we need the first input shape for this inference.
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }

  auto input_shape = ctx.getInputType(0)->tensor_type().shape();

  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.dim_size());

  std::vector<int64_t> strides;
  if (getRepeatedAttribute(ctx, "strides", strides)) {
    if (strides.size() != n_input_dims) {
      fail_shape_inference("Attribute strides has incorrect size")
    }
  } else {
    strides.assign(n_input_dims, 1);
  }

  auto *output_shape =
      ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  for (int d = 0; d < n_input_dims; d++) {
    if (strides[d] != 0) {
      *output_shape->add_dim() = (input_shape.dim(d) / strides[d]);
    } else {
      fail_shape_inference("Attribute stride of zero")
    }
  }
}

// Do the same as BatchNormalization & InstanceNormalization in onnx
void GroupNormalizationShapeInference(InferenceContext &ctx) {
  const unsigned int X_IN = 0;

  // Unused: const unsigned int Y_OUT    = 0;
  const unsigned int MEAN_OUT = 1;
  const unsigned int STD_OUT  = 2;

  const unsigned int NUM_OUTPUTS = 3;
  for (unsigned int i = 0; i < NUM_OUTPUTS; i++) {
    propagateElemTypeFromInputToOutput(ctx, X_IN, i);
  }

  propagateShapeFromInputToOutput(ctx, X_IN, 0);

  int32_t num_groups = 1;
  getAttribute(ctx, "num_groups", num_groups);

  auto input_shape = ctx.getInputType(X_IN)->tensor_type().shape();

  if (input_shape.dim_size() > 0 && input_shape.dim(0).has_dim_value()) {
    auto *output_shape_mean =
        ctx.getOutputType(MEAN_OUT)->mutable_tensor_type()->mutable_shape();
    output_shape_mean->add_dim()->set_dim_value(num_groups *
                                                input_shape.dim(0).dim_value());

    auto *output_shape_stdev =
        ctx.getOutputType(STD_OUT)->mutable_tensor_type()->mutable_shape();
    output_shape_stdev->add_dim()->set_dim_value(
        num_groups * input_shape.dim(0).dim_value());
  }
}

void PrintTensorShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void ScaleShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void LSTMShapeInference(InferenceContext &ctx) {
  const unsigned int X_IN       = 0;
  const unsigned int WEIGHTS_IN = 1;

  const unsigned int OUTPUT_OUT     = 0;
  const unsigned int CELL_STATE_OUT = 1;

  propagateElemTypeFromInputToOutput(ctx, X_IN, OUTPUT_OUT);
  propagateElemTypeFromInputToOutput(ctx, X_IN, CELL_STATE_OUT);

  // we need the first 2 input shapes for this inference.
  if (!hasNInputShapes(ctx, 2)) {
    return;
  }

  auto input_shape   = ctx.getInputType(X_IN)->tensor_type().shape();
  auto weights_shape = ctx.getInputType(WEIGHTS_IN)->tensor_type().shape();

  auto seq_length = input_shape.dim(0);
  auto batch_size = input_shape.dim(1);
  auto input_size = input_shape.dim(2);

  auto hidden_size = weights_shape.dim(2);

  int64_t output_full_sequence = 1;
  getAttribute(ctx, "output_full_sequence", output_full_sequence);

  auto *output_shape =
      ctx.getOutputType(OUTPUT_OUT)->mutable_tensor_type()->mutable_shape();
  if (output_full_sequence != 0) {
    *output_shape->add_dim() = seq_length;
  }
  *output_shape->add_dim() = batch_size;
  *output_shape->add_dim() = hidden_size;

  auto *cell_state_shape =
      ctx.getOutputType(CELL_STATE_OUT)->mutable_tensor_type()->mutable_shape();
  *cell_state_shape->add_dim() = batch_size;
  *cell_state_shape->add_dim() = hidden_size;
}

void GeluShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void GeluErfShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void DetachShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void CallShapeInference(InferenceContext &ctx) {
  auto num_inputs = ctx.getNumInputs();

  std::vector<const ONNX_NAMESPACE::TypeProto *> subgraph_input_types;

  // Run inferencing on the subgraph
  std::vector<const ONNX_NAMESPACE::TypeProto *> subgraph_output_types;
  GraphInferencer *graphInferencer = ctx.getGraphAttributeInferencer("callee");
  if (graphInferencer) {
    std::vector<const ONNX_NAMESPACE::TensorProto *> input_data;
    for (size_t i = 0; i < num_inputs; ++i) {
      input_data.push_back(ctx.getInputData(i));
      subgraph_input_types.push_back(ctx.getInputType(i));
    }

    subgraph_output_types =
        graphInferencer->doInferencing(subgraph_input_types, input_data);
  }

  // if empty(), assume inferencing was skipped
  if (!subgraph_output_types.empty()) {
    auto num_outputs = ctx.getNumOutputs();

    if (subgraph_output_types.size() != num_outputs) {
      fail_type_inference(
          "Graph attribute inferencing returned type information for ",
          subgraph_output_types.size(),
          " outputs. Expected ",
          num_outputs);
    }

    for (size_t i = 0; i < num_outputs; ++i) {
      auto *subgraph_output_type = subgraph_output_types[i];
      auto *output_type          = ctx.getOutputType(i);

      if (!subgraph_output_type->has_tensor_type()) {
        fail_type_inference(
            "Graph 'callee' subgraph outputs should all be tensors but output ",
            i,
            " was ",
            subgraph_output_type->value_case());
      }

      // if there's an existing type check it matches. otherwise propagate
      propagateElemTypeWithValidation(subgraph_output_type, output_type);

      // merge shape; type already propagated
      auto &subgraph_output_tensor_type = subgraph_output_type->tensor_type();
      auto *mutable_call_output_tensor_type =
          output_type->mutable_tensor_type();

      mergeInShapeInfo(subgraph_output_tensor_type,
                       *mutable_call_output_tensor_type);
    }
  }
}

void DynamicUpdateShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void DynamicSliceShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void DynamicZeroShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void DynamicAddShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void SequenceSliceShapeInference(InferenceContext &ctx) {
  propagateShapeFromInputToOutput(ctx, 1, 0);
  propagateElemTypeFromInputToOutput(ctx, 1, 0);
}

template <unsigned int infer_shape_index>
void LossShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  std::string reduction = getAttribute(ctx, "reduction", "Mean");

  if (reduction == "None") {
    if (hasInputShape(ctx, infer_shape_index)) {
      propagateShapeFromInputToOutput(ctx, infer_shape_index, 0);
    }
  } else {
    if (reduction != "Mean" && reduction != "Sum") {
      throw popart::internal_error("No loss reduction type for {}", reduction);
    }

    // Scalar output
    updateOutputShape(ctx, 0, TensorShapeProto());
  }
}

void MultiConvShapeInference(InferenceContext &ctx) {
  auto num_outputs              = ctx.getNumOutputs();
  int64_t cumulativeSpatialDims = 0;

  popart::Shape flatPads;
  popart::Shape flatOutPads;
  popart::Shape flatStrides;
  popart::Shape flatDilations;
  popart::Shape flatInDilations;
  getRepeatedAttribute(ctx, "pads", flatPads);
  getRepeatedAttribute(ctx, "outPads", flatOutPads);
  getRepeatedAttribute(ctx, "strides", flatStrides);
  getRepeatedAttribute(ctx, "dilations", flatDilations);
  getRepeatedAttribute(ctx, "inDilations", flatInDilations);

  for (size_t outIdx = 0; outIdx < num_outputs; ++outIdx) {
    size_t dataIdx    = (3 * outIdx);
    size_t weightsIdx = (3 * outIdx) + 1;

    propagateElemTypeFromInputToOutput(ctx, dataIdx, outIdx);

    auto dataShape    = ctx.getInputType(dataIdx)->tensor_type().shape();
    auto weightsShape = ctx.getInputType(weightsIdx)->tensor_type().shape();

    auto *outputShape =
        ctx.getOutputType(outIdx)->mutable_tensor_type()->mutable_shape();

    *outputShape->add_dim() = dataShape.dim(0);    // batch size
    *outputShape->add_dim() = weightsShape.dim(0); // num output channels

    auto dataSize    = dataShape.dim_size();
    auto spatialSize = dataSize - 2;

    // Unflatten parameters
    popart::Shape pads;
    popart::Shape outPads;
    popart::Shape strides;
    popart::Shape dilations;
    popart::Shape inDilations;
    if (flatPads.empty()) {
      pads.assign(spatialSize * 2, 0);
    } else {
      const auto cumulativePads = cumulativeSpatialDims * 2;
      pads                      = {flatPads.begin() + cumulativePads,
              flatPads.begin() + cumulativePads + (spatialSize * 2)};
    }
    if (flatOutPads.empty()) {
      outPads.assign(spatialSize * 2, 0);
    } else {
      const auto cumulativePads = cumulativeSpatialDims * 2;
      outPads                   = {flatOutPads.begin() + cumulativePads,
                 flatOutPads.begin() + cumulativePads + (spatialSize * 2)};
    }
    if (flatStrides.empty()) {
      strides.assign(spatialSize, 1);
    } else {
      strides = {flatStrides.begin() + cumulativeSpatialDims,
                 flatStrides.begin() + cumulativeSpatialDims + spatialSize};
    }
    if (flatDilations.empty()) {
      dilations.assign(spatialSize, 1);
    } else {
      dilations = {flatDilations.begin() + cumulativeSpatialDims,
                   flatDilations.begin() + cumulativeSpatialDims + spatialSize};
    }
    if (flatInDilations.empty()) {
      inDilations.assign(spatialSize, 1);
    } else {
      inDilations = {flatInDilations.begin() + cumulativeSpatialDims,
                     flatInDilations.begin() + cumulativeSpatialDims +
                         spatialSize};
    }
    cumulativeSpatialDims += spatialSize;

    popart::Shape spatialDShape;
    popart::Shape spatialKShape;
    for (int i = 2; i < dataSize; i++) {
      spatialDShape.push_back(dataShape.dim(i).dim_value());
      spatialKShape.push_back(weightsShape.dim(i).dim_value());
    }

    popart::Shape spatialOutShape =
        popart::HasReceptiveFieldOp::getSpatialOutShape(
            spatialDShape,
            spatialKShape,
            pads,
            outPads,
            strides,
            dilations,
            inDilations,
            popart::AutoPad::NOTSET);

    for (auto dim : spatialOutShape) {
      outputShape->add_dim()->set_dim_value(dim);
    }
  }
}

void NopShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void ShapedDropoutShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void Atan2ShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (!hasNInputShapes(ctx, 2)) {
    fail_type_inference("Atan2 requires two inputtensors with shapes.");
  }

  bidirectionalBroadcastShapeInference(
      ctx.getInputType(0)->tensor_type().shape(),
      ctx.getInputType(1)->tensor_type().shape(),
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void DepthToSpaceShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  auto blocksize = getAttribute(ctx, "blocksize", 0);
  if (blocksize <= 0) {
    fail_shape_inference("Blocksize must be positive");
  }

  if (hasInputShape(ctx, 0)) {
    auto &input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() == 4) {
      updateOutputShape(ctx,
                        0,
                        {input_shape.dim(0),
                         input_shape.dim(1) / (blocksize * blocksize),
                         input_shape.dim(2) * blocksize,
                         input_shape.dim(3) * blocksize});
    } else {
      fail_shape_inference("Input tensor must be 4-dimensional");
    }
  }
}

void Expm1ShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void Log1pShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void ReshapeShapeInference(InferenceContext &ctx) {
  propagateShapeFromAttributeToOutput(ctx, "shape", 0);
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
}

void ReverseShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void SliceShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  std::vector<int64_t> axes;
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;

  if (!getRepeatedAttribute(ctx, "axes", axes)) {
    fail_shape_inference("Missing axes");
  }
  if (!getRepeatedAttribute(ctx, "starts", starts)) {
    fail_shape_inference("Missing starts");
  }
  if (!getRepeatedAttribute(ctx, "ends", ends)) {
    fail_shape_inference("Missing ends");
  }

  if (axes.size() != starts.size() || axes.size() != ends.size()) {
    fail_shape_inference("Attribute size mismatch")
  }

  if (!std::is_sorted(axes.begin(), axes.end())) {
    fail_shape_inference("Axes not sorted")
  }

  ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

  const onnx::TensorShapeProto &input_shape(
      ctx.getInputType(0)->tensor_type().shape());

  // Because the axes are in ascending order, we can walk through them and the
  // input shape at the same time. This will hold the axes index.
  size_t current_axis_idx = 0;

  for (size_t idx = 0; idx < input_shape.dim_size(); idx++) {
    if (!input_shape.dim(idx).has_dim_value()) {
      fail_shape_inference("Input shape incomplete")
    }

    onnx::TensorShapeProto_Dimension *new_dim =
        ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();

    // Assume no slice on this dim to allow continue
    int64_t cur_size = input_shape.dim(idx).dim_value();
    new_dim->set_dim_value(cur_size);

    // No more axes to slice
    if (current_axis_idx == axes.size()) {
      continue;
    }

    // Not slicing on this axis
    if (axes[current_axis_idx] != idx) {
      continue;
    }

    int64_t start = starts[current_axis_idx];
    int64_t end   = ends[current_axis_idx];

    start = start < 0 ? cur_size + start : start;
    end   = end < 0 ? cur_size + end : end;
    start = std::min(start, cur_size);
    start = std::max(start, static_cast<int64_t>(0));
    end   = std::min(end, cur_size);
    end   = std::max(end, static_cast<int64_t>(0));

    if (start > end) {
      fail_shape_inference("Start greater than end for an axis");
    }

    new_dim->set_dim_value(end - start);
    current_axis_idx++;
  }
}

void InitShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromAttributeToOutput(ctx, "data_type", 0);
  propagateShapeFromAttributeToOutput(ctx, "shape", 0);
}

void ScatterReduceShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (ctx.getNumInputs() == 3) {
    // The optional initial_values argument is supplied as the third input
    propagateShapeFromInputToOutput(ctx, 2, 0);
    return;
  }

  // The output shape is same as the data source tensor.
  // Except in the scatter axis is equal to the axis_size
  const auto axis     = getIntAttribute(ctx, "axis");
  const auto axisSize = getIntAttribute(ctx, "axis_size");

  auto &inputShape  = getInputShape(ctx, 0);
  const auto rank   = inputShape.dim_size();
  auto *outputShape = getOutputShape(ctx, 0);

  for (int i = 0; i < rank; i++) {
    const auto value = i == axis ? axisSize : inputShape.dim(i).dim_value();
    outputShape->add_dim()->set_dim_value(value);
  }
}

void GroupedGatherShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  auto axis = getIntAttribute(ctx, "axis");

  const auto &inputShape   = getInputShape(ctx, 0);
  const auto &indicesShape = getInputShape(ctx, 1);
  const auto inputRank     = inputShape.dim_size();
  const auto indicesRank   = indicesShape.dim_size();

  if (inputRank == 0) {
    propagateShapeFromInputToOutput(ctx, 0, 0);
    return;
  }

  // ONNX allows the axis attribute to be negative
  axis = axis % inputRank; // axis in the range [-m+1, m-1]
  axis += inputRank;       // axis in the range [0, 2m-1]
  axis = axis % inputRank; // axis in the range [0, m-1]

  auto *outputShape = getOutputShape(ctx, 0);

  for (int i = 0; i < axis; i++) {
    outputShape->add_dim()->set_dim_value(inputShape.dim(i).dim_value());
  }

  for (int i = 1; i < indicesRank; ++i) {
    outputShape->add_dim()->set_dim_value(indicesShape.dim(i).dim_value());
  }

  for (int i = axis + 1; i < inputRank; i++) {
    outputShape->add_dim()->set_dim_value(inputShape.dim(i).dim_value());
  }
}

void BucketizeShapeInference(InferenceContext &ctx) {
  auto &&outputMutableType = ctx.getOutputType(0)->mutable_tensor_type();

  outputMutableType->set_elem_type(TensorProto::INT32);
  *outputMutableType->mutable_shape() =
      ctx.getInputType(0)->tensor_type().shape();
}

void SortShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  propagateShapeFromInputToOutput(ctx, 0, 0);

  auto &&indicesMutableType = ctx.getOutputType(1)->mutable_tensor_type();
  indicesMutableType->set_elem_type(TensorProto::INT32);
  propagateShapeFromInputToOutput(ctx, 0, 1);
}

void RemainderShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (!hasNInputShapes(ctx, 2)) {
    fail_shape_inference("Remainder requires two input tensors with shapes.");
  }

  bidirectionalBroadcastShapeInference(
      ctx.getInputType(0)->tensor_type().shape(),
      ctx.getInputType(1)->tensor_type().shape(),
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void FmodShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (!hasNInputShapes(ctx, 2)) {
    fail_shape_inference("Fmod requires two input tensors with shapes.");
  }

  bidirectionalBroadcastShapeInference(
      ctx.getInputType(0)->tensor_type().shape(),
      ctx.getInputType(1)->tensor_type().shape(),
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void BitwiseNotShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void RoundShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void CtcBeamSearchDecoderShapeInference(InferenceContext &ctx) {
  auto batchSize = getInputShape(ctx, 0).dim(1).dim_value();
  auto topPaths  = getIntAttribute(ctx, "top_paths");
  auto maxTime   = getInputShape(ctx, 0).dim(0).dim_value();

  auto *labelProbsOutputShape = getOutputShape(ctx, 0);
  labelProbsOutputShape->add_dim()->set_dim_value(batchSize);
  labelProbsOutputShape->add_dim()->set_dim_value(topPaths);

  auto *labelLengthsOutputShape = getOutputShape(ctx, 1);
  labelLengthsOutputShape->add_dim()->set_dim_value(batchSize);
  labelLengthsOutputShape->add_dim()->set_dim_value(topPaths);

  auto *ldecodedLabelsOutputShape = getOutputShape(ctx, 2);
  ldecodedLabelsOutputShape->add_dim()->set_dim_value(batchSize);
  ldecodedLabelsOutputShape->add_dim()->set_dim_value(topPaths);
  ldecodedLabelsOutputShape->add_dim()->set_dim_value(maxTime);
}

void CtcLossShapeInference(InferenceContext &ctx) {

  std::string reduction = getAttribute(ctx, "reduction", "Mean");

  // Default output type to input type (but allow override with dtype attr).
  int64_t inType = ctx.getInputType(0)->tensor_type().elem_type();
  auto dtype     = getAttribute(ctx, "dtype", inType);

  // Propagate types.
  propagateElemTypeFromDtypeToOutput(ctx, dtype, 0);
  propagateElemTypeFromDtypeToOutput(ctx, dtype, 1);

  // Propagate shapes.
  if (reduction == "None") {
    if (hasInputShape(ctx, 0)) {
      auto &input_shape = getInputShape(ctx, 0);
      if (input_shape.dim_size() == 3) {
        updateOutputShape(ctx, 0, {input_shape.dim(1)});
      } else {
        fail_shape_inference("Input tensor must be 3-dimensional");
      }
    }
  } else if (reduction == "Mean" || reduction == "Sum") {
    updateOutputShape(ctx, 0, TensorShapeProto());
  } else {
    throw popart::internal_error("No loss reduction type for {}", reduction);
  }

  // Second output shape matches first input shape irrespective of reduction.
  propagateShapeFromInputToOutput(ctx, 0, 1);
}

void ReduceMedianShapeInference(InferenceContext &ctx) {

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (!hasNInputShapes(ctx, 1)) {
    fail_shape_inference(
        "Reduce median requires one input tensor with shapes.");
  }

  const auto inputShape = getInputShape(ctx, 0);

  std::vector<int64_t> axes;
  bool has_default_axes(!getRepeatedAttribute(ctx, "axes", axes));

  if (has_default_axes) {
    // No axes, we should reduce over ALL axes.
    axes.resize(inputShape.dim_size());
    std::iota(axes.begin(), axes.end(), int64_t(0));
  } else {
    // Normalize to positive axes.
    popart::normalizeReduceAxes(axes, inputShape.dim_size());
    // Sort the axes for general backend compatibility.
    std::sort(axes.begin(), axes.end());
    // Check the axes are all in the right range.
    popart::validateReduceAxes(axes, inputShape.dim_size(), "");
  }

  auto *outputShape   = getOutputShape(ctx, 0);
  bool keepdims       = getAttribute(ctx, "keepdims", 1);
  size_t n_input_dims = static_cast<size_t>(inputShape.dim_size());

  auto output_type = ctx.getOutputType(1);
  output_type->mutable_tensor_type()->set_elem_type(TensorProto::INT32);
  auto *indicesOutputShape = getOutputShape(ctx, 1);

  for (int i = 0; i < n_input_dims; ++i) {
    if (!std::count(axes.begin(), axes.end(), i)) {
      outputShape->add_dim()->set_dim_value(inputShape.dim(i).dim_value());
      indicesOutputShape->add_dim()->set_dim_value(
          inputShape.dim(i).dim_value());
    } else if (keepdims) {
      outputShape->add_dim()->set_dim_value(1);
      indicesOutputShape->add_dim()->set_dim_value(1);
    } else {
    }
  }
}

void BidirectionalBroadcastShapeInference(InferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);

  if (!hasNInputShapes(ctx, 2)) {
    fail_shape_inference("Bitwise requires two input tensors with shapes.");
  }

  bidirectionalBroadcastShapeInference(
      ctx.getInputType(0)->tensor_type().shape(),
      ctx.getInputType(1)->tensor_type().shape(),
      *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}

void CopyVarUpdateShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

void SwishShapeInference(InferenceContext &ctx) {
  propagateShapeAndTypeFromFirstInput(ctx);
}

extern size_t dbg_count_check_BatchNormalization_AiGraphcore_ver1;
extern size_t dbg_count_check_GroupNormalization_AiGraphcore_ver1;
extern size_t dbg_count_check_Subsample_AiGraphcore_ver1;
extern size_t dbg_count_check_PrintTensor_AiGraphcore_ver1;
extern size_t dbg_count_check_Scale_AiGraphcore_ver1;
extern size_t dbg_count_check_LSTM_AiGraphcore_ver1;
extern size_t dbg_count_check_Gelu_AiGraphcore_ver1;
extern size_t dbg_count_check_GeluErf_AiGraphcore_ver1;
extern size_t dbg_count_check_Detach_AiGraphcore_ver1;
extern size_t dbg_count_check_Call_AiGraphcore_ver1;
extern size_t dbg_count_check_L1_AiGraphcore_ver1;
extern size_t dbg_count_check_DynamicUpdate_AiGraphcore_ver1;
extern size_t dbg_count_check_DynamicSlice_AiGraphcore_ver1;
extern size_t dbg_count_check_DynamicZero_AiGraphcore_ver1;
extern size_t dbg_count_check_DynamicAdd_AiGraphcore_ver1;
extern size_t dbg_count_check_SequenceSlice_AiGraphcore_ver1;
extern size_t dbg_count_check_MultiConv_AiGraphcore_ver1;
extern size_t dbg_count_check_Nop_AiGraphcore_ver1;
extern size_t dbg_count_check_ShapedDropout_AiGraphcore_ver1;
extern size_t dbg_count_check_Atan2_AiGraphcore_ver1;
extern size_t dbg_count_check_DepthToSpace_AiGraphcore_ver1;
extern size_t dbg_count_check_Expm1_AiGraphcore_ver1;
extern size_t dbg_count_check_Log1p_AiGraphcore_ver1;
extern size_t dbg_count_check_Reshape_AiGraphcore_ver1;
extern size_t dbg_count_check_Resize_AiGraphcore_ver1;
extern size_t dbg_count_check_ScatterReduce_AiGraphcore_ver1;
extern size_t dbg_count_check_Init_AiGraphcore_ver1;
extern size_t dbg_count_check_Remainder_AiGraphcore_ver1;
extern size_t dbg_count_check_Fmod_AiGraphcore_ver1;
extern size_t dbg_count_check_BitwiseNot_AiGraphcore_ver1;
extern size_t dbg_count_check_Round_AiGraphcore_ver1;
extern size_t dbg_count_check_CtcBeamSearchDecoder_AiGraphcore_ver1;
extern size_t dbg_count_check_CtcLoss_AiGraphcore_ver1;
extern size_t dbg_count_check_ReduceMedian_AiGraphcore_ver1;
extern size_t dbg_count_check_BitwiseAnd_AiGraphcore_ver1;
extern size_t dbg_count_check_BitwiseOr_AiGraphcore_ver1;
extern size_t dbg_count_check_BitwiseXor_AiGraphcore_ver1;
extern size_t dbg_count_check_BitwiseXnor_AiGraphcore_ver1;
extern size_t dbg_count_check_CopyVarUpdate_AiGraphcore_ver1;
extern size_t dbg_count_check_Swish_AiGraphcore_ver1;
extern size_t dbg_count_check_Bucketize_AiGraphcore_ver1;
extern size_t dbg_count_check_GroupedGather_AiGraphcore_ver1;
extern size_t dbg_count_check_Sort_AiGraphcore_ver1;

ONNX_OPERATOR_SET_SCHEMA_EX(

    BatchNormalization,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Unbiased batch normalization.")
        .Input(0, "x", "Input tensor", "T")
        .Input(1, "scale", "Scale", "T")
        .Input(2, "b", "Bias", "T")
        .Input(3, "mean", "Mean", "T")
        .Input(4, "var", "Variance", "T")
        .Output(0, "Y", "Output tensor", "T")
        .Output(1, "Mean", "The mean after GroupNormalization operator", "T")
        .Output(2, "Var", "The variance after GroupNormalization operator", "T")
        .Output(3,
                "SavedMean",
                "The variance after GroupNormalization operator",
                "T")
        .Output(4,
                "SavedVar",
                "The variance after GroupNormalization operator",
                "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .Attr("num_outputs", "The number of groups", AttributeProto::INT, false)
        .Attr("epsilon", "Epsilon", AttributeProto::FLOAT, 1e-5f)
        .Attr("momentum", "Momentum", AttributeProto::FLOAT, 0.9f)
        .Attr("unbiased_variance", "", AttributeProto::INT, false))

static const char groupnormalizationDoc[] =
    "GroupNormalization applies Group Normalization over a mini-batch of "
    "input";

ONNX_OPERATOR_SET_SCHEMA_EX(

    GroupNormalization,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc(groupnormalizationDoc)
        .Input(0, "X", "Input tensor", "T")
        .Input(1,
               "Scale",
               "The input 1-dimensional scale tensor of size C.",
               "T")
        .Input(2, "Bias", "The input 1-dimensional bias tensor of size C.", "T")
        .Output(0, "Y", "Output tensor", "T")
        .Output(1, "Mean", "The mean after GroupNormalization operator", "T")
        .Output(2, "Var", "The variance after GroupNormalization operator", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("num_groups", "The number of groups", AttributeProto::INT, false)
        .Attr("epsilon",
              "The epsilon value to use to avoid division by zero.",
              AttributeProto::FLOAT,
              1e-5f)
        .TypeAndShapeInferenceFunction(GroupNormalizationShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Subsample,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Subsample takes every Nth element of a tensor.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("strides",
              "Strides in each of the dimensions.",
              AttributeProto::INTS,
              false)
        .TypeAndShapeInferenceFunction(SubsampleShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    PrintTensor,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("PrintTensor prints the value of a tensor.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("print_gradient",
              "Should the gradient tensor also be printed.",
              AttributeProto::INT,
              true)
        .Attr("title",
              "A sting to use for the title of the tensor. Will use tensor "
              "name if not provided.",
              AttributeProto::STRING,
              "")
        .TypeAndShapeInferenceFunction(PrintTensorShapeInference))

static const char scaleDoc[] =
    "Scale takes one input data (Tensor<float>) and produces one output data "
    "(Tensor<float>) whose value is the input data tensor scaled "
    "element-wise.";

ONNX_OPERATOR_SET_SCHEMA_EX(

    Scale,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc(scaleDoc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("scale", "The scale to apply", AttributeProto::FLOAT, true)
        .TypeAndShapeInferenceFunction(ScaleShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    LSTM,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("")
        .Input(0, "X", "The input tensor", "T")
        .Input(1, "Weights", "The concatenated input and output weights", "T")
        .Input(2, "Bias", "The biases", "T")
        .Input(3, "InitState", "The initial state", "T")
        .Output(0, "Output", "Output tensor", "T")
        .Output(1, "CellState", "The lstm cell state", "T")
        // Optional (training) output 2 ignored
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .Attr("output_full_sequence",
              "If true, the lstm returns the entire sequence of outputs, "
              "otherwise it just returns the final output.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction(LSTMShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Gelu,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Applies the Gaussian Error Linear Units function.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(GeluShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    GeluErf,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Applies the accurate Gaussian Error Linear Units using Error "
                "Function.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(GeluErfShapeInference))

static const char detachDoc[] =
    "An IdentityOp that doesn't return any grad ops. This allows you to "
    "disconnect the flow of gradients when creating the backwards pass";

ONNX_OPERATOR_SET_SCHEMA_EX(

    Detach,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc(detachDoc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)",
                         "tensor(int32)",
                         "tensor(float16)",
                         "tensor(bool)"},
                        "Do not constrain tensors")
        .TypeAndShapeInferenceFunction(DetachShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Call,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Call instantiates a call to a subgraph.")
        .Input(0,
               "inputs",
               "List of inputs to the subgraph",
               "T",
               OpSchema::Variadic)
        .Output(0,
                "outputs",
                "List of outputs from the subgraph",
                "T",
                OpSchema::Variadic)
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("callee",
              "The subgraph to call into.",
              AttributeProto::GRAPH,
              true)
        .TypeAndShapeInferenceFunction(CallShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    DynamicUpdate,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Update a subtensor at offsets read from a tensor.")
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "offset", "Offset tensor", "T")
        .Input(2, "slice", "Slice tensor", "T")
        .Output(0, "output", "TensorId of the output", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("axes", "Axes along which to update.", AttributeProto::INTS, true)
        .Attr("size",
              "Size of the slice in each axis.",
              AttributeProto::INTS,
              true)
        .Attr("noOverlap", ".", AttributeProto::INT, true)
        .TypeAndShapeInferenceFunction(DynamicUpdateShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    DynamicSlice,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Slice a tensor based on offsets specified by a tensor.")
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "index", "Index tensor", "T")
        .Output(0, "output", "TensorId of the output", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("axes", "Axes along which to update.", AttributeProto::INTS, true)
        .Attr("size",
              "Size of the slice in each axis.",
              AttributeProto::INTS,
              true)
        .Attr("noOverlap", ".", AttributeProto::INT, true)
        .TypeAndShapeInferenceFunction(DynamicSliceShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    DynamicZero,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Creates a copy of 'tensor' with a slice at 'offset' set to "
                "zero, e.g. out = tensor, out[offset] = 0.0")
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "offset", "Offset tensor", "T")
        .Input(2, "slice", "Slice tensor", "T")
        .Output(0, "output", "TensorId of the output", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("axes", "Axes along which to erase.", AttributeProto::INTS, true)
        .Attr("size",
              "Size of the slice in each axis.",
              AttributeProto::INTS,
              true)
        .TypeAndShapeInferenceFunction(DynamicZeroShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    DynamicAdd,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Creates a copy of 'tensor' with 'slice' added at 'offset', "
                "e.g. out = tensor, out[offset] += slice")
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "offset", "Offset tensor", "T")
        .Input(2, "slice", "Slice tensor", "T")
        .Output(0, "output", "TensorId of the output", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)"},
            "Constrain input and output types to signed numeric tensors.")
        .Attr("axes", "Axes along which to add.", AttributeProto::INTS, true)
        .Attr("size",
              "Size of the slice in each axis.",
              AttributeProto::INTS,
              true)
        .TypeAndShapeInferenceFunction(DynamicAddShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    SequenceSlice,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Slice a 2d tensor based on offsets specified by a tensor.")
        .Input(0, "source", "Source tensor", "T")
        .Input(1, "destination", "Destination tensor", "T")
        .Input(2, "N", "The number of elements to copy", "T")
        .Input(3, "sourceOffset", "First element from source to read from", "T")
        .Input(4,
               "destinationOffset",
               "First element in destination to wite to",
               "T")
        .Output(0, "output", "TensorId of the output", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("zeroUnused",
              "Zero unreference elements",
              AttributeProto::INT,
              true)
        .TypeAndShapeInferenceFunction(SequenceSliceShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    L1,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Calculates the absolute values of each element in the input "
                "and optionally mean/sum reduce the output")
        .Input(
            0,
            "input",
            "Any shape tensor for which to calculate to calculate the scaled "
            "absolute values or l1 norm",
            "T")
        .Output(0, "output", "The scaled absolute values / l1 norm", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(float16)",
             "tensor(uint32)",
             "tensor(int32)"},
            "Constrain input and output types to float(32/16) and (u)int32 "
            "tensors.")
        .Attr("lambda",
              "Scaling factor for the loss",
              AttributeProto::FLOAT,
              true)
        .Attr("reduction",
              "Reduction type: None, Mean or Sum",
              AttributeProto::STRING)
        .TypeAndShapeInferenceFunction(LossShapeInference<0>))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Nll,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Calculates the negative likelihood loss on based on the "
                "probability and label inputs")
        .Input(0, "probs", "Tensor of shape [D1, ..., DN, NumClasses]", "T")
        .Input(1,
               "label",
               "Tensor of shape [D1, ..., DN] where each element is a class "
               "index",
               "Tind")
        .Output(0,
                "output",
                "The negative log likelihood loss, possibly "
                "sum/mean reduced",
                "T")
        .TypeConstraint("T",
                        {"tensor(float16)", "tensor(float)"},
                        "Floating Point Tensors")
        .TypeConstraint("Tind",
                        {"tensor(int16)", "tensor(int32)"},
                        "Integer types")
        .Attr("reduction",
              "Reduction type: None, Mean or Sum",
              AttributeProto::STRING,
              true)
        .Attr("ignoreIndex",
              "If non negative, ignores targets with the same index so that "
              "they do not contribute to the loss",
              AttributeProto::INT,
              false)
        .Attr("inputIsLogProbability",
              "Specifies if the input tensor contains log-probabilities or raw "
              "probabilities",
              AttributeProto::INT,
              false)
        .TypeAndShapeInferenceFunction(LossShapeInference<1>))

ONNX_OPERATOR_SET_SCHEMA_EX(

    IdentityLoss,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Outputs the identity of the inputs, or the sum or mean "
                "of the inputs: generally used as a utility to provide the "
                "reduction of other losses")
        .Input(0, "input", "Tensor inputs to the loss", "T")
        .Output(0,
                "outputs",
                "A Tensor (identical to input) or scalar output depending on "
                "whether reduction is specified",
                "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)",
             "tensor(float)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("reduction",
              "Reduction type: None, Mean or Sum",
              AttributeProto::STRING,
              true)
        .TypeAndShapeInferenceFunction(LossShapeInference<0>))

ONNX_OPERATOR_SET_SCHEMA_EX(

    MultiConv,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("MultiConv comprises multiple convolution operations that can "
                "be run in parallel")
        .Input(0,
               "inputs",
               "List of inputs to the convolutions, in triplets 'data, "
               "weights, biases' order",
               "T",
               OpSchema::Variadic)
        .Output(0,
                "outputs",
                "List of outputs from the convolutions",
                "T",
                OpSchema::Variadic)
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("strides",
              "The concatenated strides for each convolution.",
              AttributeProto::INTS,
              false)
        .Attr("pads",
              "The concatenated pads for each convolution.",
              AttributeProto::INTS,
              false)
        .Attr("dilations",
              "The concatenated dilations for each convolution.",
              AttributeProto::INTS,
              false)
        .Attr("planType", "The plan type.", AttributeProto::STRING, false)
        .Attr("perConvReservedTiles",
              "The number of tiles reserved per convolution.",
              AttributeProto::INT,
              false)
        .Attr("perConvReservedTiles",
              "The number of tiles reserved per convolution.",
              AttributeProto::INT,
              false)
        .Attr("cycleBackOff",
              "The cycle back-off proportion.",
              AttributeProto::FLOAT,
              false)
        .Attr(popart::sAvailMemAttribute,
              "The available memory proportion per convolution.",
              AttributeProto::FLOATS,
              false)
        .Attr(popart::sPartialsTypeAttribute,
              "The partials type when computing each convolution.",
              AttributeProto::STRINGS,
              false)
        .Attr("numConvs",
              "The number convolutions that the MultiConv comprises.",
              AttributeProto::INT,
              true)
        .TypeAndShapeInferenceFunction(MultiConvShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Nop,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("No operation, for debugging purposes.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(int32)", "tensor(float16)"},
            "Constrain input and output types to signed numeric tensors.")
        .TypeAndShapeInferenceFunction(NopShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    ShapedDropout,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Applies a shaped dropout to the input tensor.")
        .Input(0, "input", "Tensor to apply shaped dropout to", "T")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(float16)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("shape",
              "Shape of dropout mask. Must be broadcastable to the input",
              AttributeProto::INTS,
              true)
        .Attr("ratio",
              "Probability of dropping an input feature (default = 0.5)",
              AttributeProto::FLOAT,
              0.5f)
        .TypeAndShapeInferenceFunction(ShapedDropoutShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Atan2,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Atan2(Y, X)")
        .Input(0, "Y", "First input tensor", "T")
        .Input(1, "X", "Second input tensor", "T")
        .Output(0, "Theta", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(Atan2ShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    DepthToSpace,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Rearrange tensor so that the spatial dimensions are scaled and"
                " the depth diemnsion is reduced.")
        .Input(0, "input", "Input tensor", "T")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint("T",
                        OpSchema::all_tensor_types(),
                        "Allow all tensor types")
        .TypeAndShapeInferenceFunction(DepthToSpaceShapeInference)
        .Attr("blocksize",
              "Blocks of [blocksize, blocksize] are moved.",
              AttributeProto::INT)
        .Attr("mode",
              "DCR (default) for depth-column-row order re-arrangement. Use "
              "CRD for column-row-depth order.",
              AttributeProto::STRING,
              std::string("DCR")))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Expm1,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Exp(x) - 1.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(Expm1ShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Log1p,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Log(x + 1).")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(Log1pShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Reshape,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Reshape the input tensor.")
        .Input(0, "data", "Input tensor", "T")
        .Output(0, "reshaped", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("shape", "New shape.", AttributeProto::INTS, true)
        .TypeAndShapeInferenceFunction(ReshapeShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Reverse,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Reverse the input tensor.")
        .Input(0, "data", "Input tensor", "T")
        .Output(0, "reversed", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("dimensions",
              "Dimensions to reverse.",
              AttributeProto::INTS,
              true)
        .TypeAndShapeInferenceFunction(ReverseShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    Slice,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Slice the input tensor.")
        .Input(0, "data", "Input tensor", "T")
        .Output(0, "sliced", "Output Tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("axes", "Axes for start and end.", AttributeProto::INTS, true)
        .Attr("starts",
              "Starting indices of corresponding axis in `axes`",
              AttributeProto::INTS,
              true)
        .Attr("ends",
              "Ending indices (exclusive) of corresponding axis in axes`",
              AttributeProto::INTS,
              true)
        .TypeAndShapeInferenceFunction(SliceShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    GroupedScatterReduce,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Reduces all the values from the input data tensor at the "
                "specified indices along the given axis for each group.")
        .Input(0, "data", "Input tensor", "T")
        .Input(1, "indices", "Indices defining the scatter operation", "T")
        .Input(2,
               "initial_values",
               "Optional values used to initialise the output tensor",
               "T")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("axis_size",
              "The size of the output in the scatter axis.",
              AttributeProto::INT,
              true)
        .Attr("axis",
              "axis to apply the scatter reduction on (default = -1)",
              AttributeProto::INT,
              static_cast<int64_t>(-1))
        .Attr("reduction",
              "Reduction applied to the scatter operation (default = \"sum\")",
              AttributeProto::STRING,
              "sum")
        .Attr("group_size",
              "The group size (default = 1) of the data.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("enable_index_broadcast",
              "Boolean flag (default = 1). If `1` index will be broadcasted to "
              "match `data` tensor size, otherwise (`0`) its size will "
              "remain unchanged.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction(ScatterReduceShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    ScatterReduce,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Reduces all the values from the input data tensor at the "
                "specified indices along the given axis.")
        .Input(0, "data", "Input tensor", "T")
        .Input(1, "indices", "Indices defining the scatter operation", "T")
        .Input(2,
               "initial_values",
               "Optional values used to initialise the output tensor",
               "T")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("axis_size",
              "The size of the output in the scatter axis.",
              AttributeProto::INT,
              true)
        .Attr("axis",
              "axis to apply the scatter reduction on (default = -1)",
              AttributeProto::INT,
              static_cast<int64_t>(-1))
        .Attr("reduction",
              "Reduction applied to the scatter operation (default = \"sum\")",
              AttributeProto::STRING,
              "sum")
        .Attr("enable_index_broadcast",
              "Boolean flag (default = 1). If `1` index will be broadcasted to "
              "match `data` tensor size, otherwise (`0`) its size will "
              "remain unchanged.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction(ScatterReduceShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    GroupedGather,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Given data tensor of rank r >= 1, and indices tensor of rank "
                "q, gather entries of the axis dimension of data"
                "(by default outer-most one as axis=0) indexed by indices, "
                "and concatenates them in an output tensor of rank "
                "q + (r - 1) for each group.")
        .Input(0, "data", "Input tensor", "T")
        .Input(1, "indices", "Indices defining the gather operation", "T")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("axis",
              "(default = 0 Which axis to gather on. Negative value means  "
              "dcounting imensions from the back. Accepted range is [-r, r-1] "
              "where r = rank(data))",
              AttributeProto::INT,
              static_cast<int64_t>(-1))
        .Attr("group_size",
              "The group size (default = 1) of the data.",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction(GroupedGatherShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Init,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc(
            "Initialize a new tensor given shape, data type, tensor type and "
            "initialization type. Allows to create initialized tensors in any "
            "graph. The InitOp has no tensor inputs and one tensor output.")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(uint8)",
             "tensor(uint16)",
             "tensor(uint32)",
             "tensor(uint64)",
             "tensor(int8)",
             "tensor(int16)",
             "tensor(int32)",
             "tensor(int64)",
             "tensor(float16)",
             "tensor(float)",
             "tensor(bool)"},
            "Input and output types can be any type supported by the IPU.")
        .Attr("shape", "The shape of the output.", AttributeProto::INTS, true)
        .Attr("data_type", "", AttributeProto::INT)
        .Attr("tensor_type", "", AttributeProto::INT)
        .Attr("init_type", "", AttributeProto::INT)
        .Attr("batch_axis", "", AttributeProto::INT)
        .TypeAndShapeInferenceFunction(InitShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Remainder,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Remainder(A, B)")
        .Input(0, "A", "Dividend tensor", "T")
        .Input(1, "B", "Divisor tensor", "T")
        .Output(0, "C", "Remainder tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)",
                         "tensor(float16)",
                         "tensor(int32)",
                         "tensor(uint32)"},
                        "Constrain input and output types to float(32/16) and "
                        "(u)int32 tensors.")
        .TypeAndShapeInferenceFunction(RemainderShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Fmod,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Fmod(A, B)")
        .Input(0, "A", "Dividend tensor", "T")
        .Input(1, "B", "Divisor tensor", "T")
        .Output(0, "C", "Remainder tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)",
                         "tensor(float16)",
                         "tensor(int32)",
                         "tensor(uint32)"},
                        "Constrain input and output types to float(32/16) and "
                        "(u)int32 tensors.")
        .TypeAndShapeInferenceFunction(FmodShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    BitwiseNot,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("BitwiseNot(X)")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(int32)", "tensor(uint32)"},
                        "Constrain input and output types to (u)int32 tensors.")
        .TypeAndShapeInferenceFunction(BitwiseNotShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Round,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Round(X)")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(float16)"},
            "Constrain input and output types to float(32/16) tensors.")
        .TypeAndShapeInferenceFunction(RoundShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    CtcBeamSearchDecoder,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Performs beam search decoding on the log probabilities given "
                "in input.")
        .Input(0, "log_probs", "Input log probabilities", "T1")
        .Input(1, "data_lengths", "Length of each input", "T2")
        .Output(0, "label_probs", "Label probabilities", "T1")
        .Output(1, "label_lengths", "Length of each output", "T2")
        .Output(2, "decoded_labels", "Decoded labels", "T2")
        .TypeConstraint("T1",
                        {
                            "tensor(float16)",
                            "tensor(float)",
                        },
                        "Input log probabilities and output label "
                        "probabilities can be float or half.")
        .TypeConstraint("T2",
                        {
                            "tensor(uint32)",
                        },
                        "Input and output lengths and decoded labels can be "
                        "unsigned 32-bit integers.")
        .Attr("blank",
              "The integer representing the blank class.",
              AttributeProto::INT)
        .Attr("beam_width",
              "The number of beams to use when decoding.",
              AttributeProto::INT)
        .Attr("top_paths",
              "The number of most likely decoded paths to return, must be less "
              "than or equal to beamWidth.",
              AttributeProto::INT)
        .TypeAndShapeInferenceFunction(CtcBeamSearchDecoderShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Ctc,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Outputs the connectionist temporal classification (CTC) loss, "
                "optionally reduced by either sum or mean")
        .Input(0,
               "log_probs",
               "Logarithmized probabilities input sequences",
               "T1")
        .Input(1, "targets", "Target sequences tensor", "T2")
        .Input(2, "input_lengths", "Input sequences lengths", "T2")
        .Input(3, "target_lengths", "Target sequences lengths", "T2")
        .Output(0,
                "ctc_loss",
                "The CTC losses tensor (optionally reduced)",
                "T1")
        .Output(1,
                "log_probs_gradient_wrt_ctc_loss",
                "The gradient of the input tensor",
                "T1")
        .TypeConstraint("T1",
                        {"tensor(float16)", "tensor(float)"},
                        "Operand must be either a FLOAT or FLOAT16 tensor")
        .TypeConstraint("T2",
                        {"tensor(uint32)"},
                        "Operand must be a UINT32 tensor")
        .Attr("reduction",
              "Reduction type: None, Mean or Sum",
              AttributeProto::STRING,
              true)
        .Attr("blank",
              "The integer value to use to represent 'no class'",
              AttributeProto::INT,
              true)
        .Attr("dtype",
              "If set, the ctc_loss output tensor assumes this data type. "
              "If unset, ctc_loss' data type is inferred from log_probs.",
              AttributeProto::INT,
              false)
        .Attr("zeroInfinity",
              "If set, infinite losses and the associated gradients are "
              "zeroed-out",
              AttributeProto::INT,
              false)
        .TypeAndShapeInferenceFunction(CtcLossShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    ReduceMedian,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("ReduceMedian(data)")
        .Input(0, "data", "Data tensor", "T")
        .Output(0, "reduced", "Reduced tensor", "T")
        .Output(1, "indices", "Indices tensor", "T2")
        .TypeConstraint("T",
                        {"tensor(uint32)",
                         "tensor(uint64)",
                         "tensor(int32)",
                         "tensor(int64)",
                         "tensor(float16)",
                         "tensor(float)"},
                        "Constrain input and output reduced tensor types to "
                        "uint(16/32), int(16/32), float(16/32) tensors.")
        .TypeConstraint("T2",
                        {"tensor(int32)"},
                        "Constrain indices tensor types to int32 tensor.")
        .Attr("axes", "The reduction axes", AttributeProto::INTS, false)
        .Attr("keepdims",
              "If to keep or squeeze reduction axes",
              AttributeProto::INT,
              true)
        .TypeAndShapeInferenceFunction(ReduceMedianShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    BitwiseAnd,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("BitwiseAnd(X, Y)")
        .Input(0, "X", "First input tensor", "T")
        .Input(1, "Y", "Second input tensor", "T")
        .Output(0, "Z", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(int32)", "tensor(uint32)"},
                        "Constrain input and output types to (u)int32 tensors.")
        .TypeAndShapeInferenceFunction(BidirectionalBroadcastShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    BitwiseOr,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("BitwiseOr(X, Y)")
        .Input(0, "X", "First input tensor", "T")
        .Input(1, "Y", "Second input tensor", "T")
        .Output(0, "Z", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(int32)", "tensor(uint32)"},
                        "Constrain input and output types to (u)int32 tensors.")
        .TypeAndShapeInferenceFunction(BidirectionalBroadcastShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    BitwiseXor,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("BitwiseXor(X, Y)")
        .Input(0, "X", "First input tensor", "T")
        .Input(1, "Y", "Second input tensor", "T")
        .Output(0, "Z", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(int32)", "tensor(uint32)"},
                        "Constrain input and output types to (u)int32 tensors.")
        .TypeAndShapeInferenceFunction(BidirectionalBroadcastShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    BitwiseXnor,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("BitwiseXnor(X, Y)")
        .Input(0, "X", "First input tensor", "T")
        .Input(1, "Y", "Second input tensor", "T")
        .Output(0, "Z", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(int32)", "tensor(uint32)"},
                        "Constrain input and output types to (u)int32 tensors.")
        .TypeAndShapeInferenceFunction(BidirectionalBroadcastShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    CopyVarUpdate,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("CopyVarUpdate(X, Y)")
        .Input(0, "X", "Var to update", "T")
        .Input(1, "Y", "Tensor to copy", "T")
        .Output(0, "Z", "Reference to updated tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)",
                         "tensor(int32)",
                         "tensor(float16)",
                         "tensor(bool)"},
                        "Do not constrain tensors")
        .TypeAndShapeInferenceFunction(CopyVarUpdateShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(

    Swish,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("Applies the Swish activation function.")
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T",
                        {"tensor(float)", "tensor(float16)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(SwishShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    Bucketize,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("The operation returns the indices of the buckets to which "
                "each value in the input tensor belongs. The ranges of each "
                "bucket are defined by the boundaries tensor. The returned "
                "index satisfies the following rules:\n"
                "right == 1: boundaries[i-1] <= input[m][n]...[l][x] "
                "< boundaries[i]\n"
                "right == 0: boundaries[i-1] < input[m][n]...[l][x] "
                "<= boundaries[i]\n")
        .Input(0,
               "input",
               "N-D input tensor or a Scalar containing the search value(s).",
               "T")
        .Input(1,
               "boundaries",
               "1-D tensor defining ranges of the buckets. "
               "This must contain a monotonically increasing sequence."
               "sequence.",
               "T")
        .Output(0,
                "out",
                "The output tensor, must be the same size and shape as input "
                "if provided.",
                "tensor(int32)")
        .TypeConstraint("T",
                        {"tensor(uint32)",
                         "tensor(int32)",
                         "tensor(float16)",
                         "tensor(float)"},
                        "Constrain input and boundaries types to float, "
                        "float16, int32 and uint32.")
        .Attr("right",
              "If 0 (default) then the left boundary is closed.",
              AttributeProto::INT,
              true)
        .TypeAndShapeInferenceFunction(BucketizeShapeInference))

ONNX_OPERATOR_SET_SCHEMA_EX(
    Sort,
    AiGraphcore,
    popart::Domain::ai_graphcore,
    1,
    false,
    OpSchema()
        .SetDoc("The operation sorts input tensor along given axis.")
        .Input(0, "input", "The input tensor", "T")
        .Output(0, "values", "The sorted values.", "T")
        .Output(1,
                "indices",
                "The indices of the elements in the original input tensor.",
                "Tind")
        .TypeConstraint("T",
                        {"tensor(uint16)",
                         "tensor(uint32)",
                         "tensor(uint64)",
                         "tensor(int8)",
                         "tensor(int16)",
                         "tensor(int32)",
                         "tensor(int64)",
                         "tensor(float16)",
                         "tensor(float)"},
                        "Constrain input and output types.")
        .TypeConstraint("Tind",
                        {"tensor(uint32)",
                         "tensor(int32)",
                         "tensor(int32)",
                         "tensor(int64)"},
                        "Constrain indices types.")
        .Attr("axis",
              "The dimension to sort along.",
              AttributeProto::INT,
              int64_t(-1))
        .Attr(
            "descending",
            "If '1' then the elements are sorted in descending order by value.",
            AttributeProto::INT,
            false)
        .Attr("stable",
              "If '1' then the sorting routine becomes stable, preserving the "
              "order of equivalent elements",
              AttributeProto::INT,
              false)
        .TypeAndShapeInferenceFunction(SortShapeInference))

static bool registerOps() {
  auto &d = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  d.AddDomainToVersion(popart::Domain::ai_graphcore, 1, 1);

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, BatchNormalization)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, GroupNormalization)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Subsample)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, PrintTensor)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Scale)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, LSTM)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Gelu)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, GeluErf)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Detach)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Call)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, L1)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Nll)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, IdentityLoss)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, MultiConv)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Nop)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, ShapedDropout)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Atan2)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, DepthToSpace)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Expm1)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Log1p)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Reshape)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Reverse)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Slice)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, ScatterReduce)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, DynamicSlice)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, DynamicAdd)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, DynamicZero)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, DynamicUpdate)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Init)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Remainder)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Fmod)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, BitwiseNot)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Round)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, CtcBeamSearchDecoder)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Ctc)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, ReduceMedian)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, BitwiseAnd)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, BitwiseOr)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, BitwiseXor)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, BitwiseXnor)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, CopyVarUpdate)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Swish)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, Bucketize)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
          AiGraphcore, 1, GroupedGather)>());

  ONNX_NAMESPACE::RegisterSchema(
      GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(AiGraphcore, 1, Sort)>());

  return true;
}

static bool ret = registerOps();

} // namespace ONNX_NAMESPACE
