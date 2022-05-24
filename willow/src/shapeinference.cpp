// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <popart/shapeinference.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

ShapeInferenceContext::ShapeInferenceContext(
    const std::map<int, TensorInfo> &inputInfos_,
    const Attributes &attributes_,
    int outputSize_)
    : inputInfos(inputInfos_), attributes(attributes_),
      outputSize(outputSize_) {}

const TensorInfo &ShapeInferenceContext::inInfo(int index) const {
  return inputInfos.at(index);
}

const Shape &ShapeInferenceContext::inShape(int index) const {
  return inputInfos.at(index).shape();
}

const Shape &ShapeInferenceContext::outShape(int index) const {
  return outputInfos.at(index).shape();
}

DataType ShapeInferenceContext::inType(int index) const {
  return inputInfos.at(index).dataType();
}

const Attributes &ShapeInferenceContext::getAttributes() const {
  return attributes;
}

TensorInfo &ShapeInferenceContext::outInfo(int index) {
  if (outputInfos.find(index) == outputInfos.end()) {
    outputInfos.insert({index, {DataType::UNDEFINED, {}}});
  }
  return outputInfos.at(index);
}

const std::map<int, TensorInfo> &ShapeInferenceContext::getOutputInfos() {
  return outputInfos;
}

int ShapeInferenceContext::getNumInputs() const { return inputInfos.size(); }

int ShapeInferenceContext::getNumOutputs() const { return outputSize; }

bool hasShape(const TensorInfo &tinfo) { return !tinfo.shape().empty(); }

bool ShapeInferenceContext::hasInputShape(size_t n) const {
  return getNumInputs() > static_cast<size_t>(n) && hasShape(inInfo(n));
}

bool ShapeInferenceContext::hasNInputShapes(size_t n) const {
  for (size_t i = 0; i < n; i++) {
    if (!hasInputShape(i)) {
      return false;
    }
  }
  return true;
}

bool ShapeInferenceContext::hasAttribute(const std::string &key) const {
  return attributes.hasAttribute(key);
}

ShapeInferenceFunctions &ShapeInferenceFunctions::getInstance() {
  static ShapeInferenceFunctions instance;
  return instance;
}

void ShapeInferenceFunctions::registerFunction(const OperatorIdentifier &opid,
                                               ShapeInferenceFunction func) {
  auto &instance = getInstance();
  if (instance.functions.find(opid) != instance.functions.end()) {
    throw error("A shape inference function has already been registered for {}",
                opid);
  }
  instance.functions.insert({opid, func});
}

bool ShapeInferenceFunctions::hasFunction(const OperatorIdentifier &opid) {
  auto &x = getInstance();
  return x.functions.find(opid) != x.functions.end();
}

ShapeInferenceFunction
ShapeInferenceFunctions::getFunction(const OperatorIdentifier &opid) {
  return getInstance().functions.at(opid);
}

RegisterShapeInferenceFunction::RegisterShapeInferenceFunction(
    const OperatorIdentifier &opid,
    ShapeInferenceFunction func) {
  ShapeInferenceFunctions::registerFunction(opid, func);
}

auto batchnormShapeInferenceFun = [](popart::ShapeInferenceContext &ctx) {
  const unsigned int X_IN           = 0;
  const unsigned int MEAN_IN        = 3;
  const unsigned int VAR_IN         = 4;
  const unsigned int Y_OUT          = 0;
  const unsigned int MEAN_OUT       = 1;
  const unsigned int VAR_OUT        = 2;
  const unsigned int SAVED_MEAN_OUT = 3;
  const unsigned int SAVED_VAR_OUT  = 4;
  propagateElemTypeFromInputToOutput(ctx, X_IN, Y_OUT);
  propagateShapeFromInputToOutput(ctx, X_IN, Y_OUT);

  const unsigned int NUM_OUT = 5;
  auto num_outputs           = ctx.getNumOutputs();
  if (num_outputs == NUM_OUT) {
    propagateElemTypeFromInputToOutput(ctx, MEAN_IN, MEAN_OUT);
    propagateElemTypeFromInputToOutput(ctx, VAR_IN, VAR_OUT);
    propagateElemTypeFromInputToOutput(ctx, MEAN_IN, SAVED_MEAN_OUT);
    propagateElemTypeFromInputToOutput(ctx, VAR_IN, SAVED_VAR_OUT);
    propagateShapeFromInputToOutput(ctx, MEAN_IN, MEAN_OUT);
    propagateShapeFromInputToOutput(ctx, VAR_IN, VAR_OUT);
    propagateShapeFromInputToOutput(ctx, MEAN_IN, SAVED_MEAN_OUT);
    propagateShapeFromInputToOutput(ctx, VAR_IN, SAVED_VAR_OUT);
  }
};

// This is an implementation of the Split-13 shape which fixes shape inference
// for Split with split attribute described in
// https://github.com/onnx/onnx/commit/dfa4384c3f8fe9d804032f992e6f818fc54152b0
// Without using this, the popart builder will be unable to infer the shape of
// its descendants.
auto splitShapeInferenceFun = [](popart::ShapeInferenceContext &ctx) {
  // Copy the data type and shape from the input tensor to the output tensors
  // (the shape of the output will not be the same as the input [we are after
  // all splitting], however, this will be addressed below)
  for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
    propagateElemTypeFromInputToOutput(ctx, 0 /* inIndex */, i /* outIndex */);
  }
  // The output shape cannot be inferred if the input tensor has no shape
  if (!ctx.hasNInputShapes(1)) {
    return;
  }

  // Get the tensor rank and the axis to make the split
  const auto &shape = ctx.inShape(0);
  int rank          = shape.size();
  int axis          = ctx.getAttribute<int64_t>("axis");

  // Check that axis is within the accepted [-rank, rank-1] range
  if (axis < -rank || axis >= rank) {
    throw popart::error(
        "Invalid value of attribute 'axis'. Rank={} Value={}", rank, axis);
  }

  // If the axis is negative, convert it to the corresponding positive index
  if (axis < 0) {
    axis += rank;
  }

  // If we are trying to split along an axis with zero dim
  const auto &split_dim = shape.at(axis);
  if (!split_dim) {
    for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
      // Update with the shape of the inTensor (rather than the outTensor)
      auto tinfo     = TensorInfo(ctx.outInfo(i).dataType(), shape);
      ctx.outInfo(i) = tinfo;
    }
    return;
  }

  std::vector<int64_t> split;
  if (ctx.hasAttribute("split")) {
    // Handle the case where the split attribute has been set
    split = ctx.getAttribute<std::vector<int64_t>>("split");
    if (split.size() != ctx.getNumOutputs()) {
      throw popart::error(
          "Mismatch between number of splits ({}) and outputs ({})",
          split.size(),
          ctx.getNumOutputs());
    }
    // Check that sum(split) = split_dim (if not we would either have too many
    // or too few indices along the axis we try to split)
    int64_t total_dim = 0;
    for (int64_t d : split) {
      total_dim += d;
    }
    if (total_dim != split_dim) {
      throw popart::error("Mismatch between the sum of 'split' ({}) and the "
                          "split dimension of the input ({})",
                          total_dim,
                          split_dim);
    }
  } else {
    // If the split attribute has not been set we try to split evenly amongst
    // the number of outputs
    int num_outputs = static_cast<int>(ctx.getNumOutputs());
    if (split_dim % num_outputs != 0) {
      throw popart::error("The input is not evenly splittable");
    }
    int chunk_size = split_dim / num_outputs;
    for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
      split.push_back(chunk_size);
    }
  }
  for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
    auto outputShape     = shape;
    outputShape.at(axis) = split.at(i);
    auto tinfo           = TensorInfo(ctx.outInfo(i).dataType(), outputShape);
    ctx.outInfo(i)       = tinfo;
  }
};

auto tensorRemapShapeInferenceFun = [](popart::ShapeInferenceContext &ctx) {
  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  propagateShapeFromInputToOutput(ctx, 0, 0);
};

// Shape inference is the process of infering the shape of the outputs of
// a node in the ONNX graph. It is run each time a new node is added to the
// model. It is executed inside BuilderImpl::runShapeInference via one of two
// code paths, depending how the shape inference function is registered:
// 1. registered in an Onnx operator schema. We do this for all aiGraphcore
//    and aiOnnx operators except for Batchnorm. See willow/src/defs.cc for
//    examples.
// 2. registered with the popart ShapeInferenceContext. This may be preferable
//    for developers of custom operators, as it does not require understanding
//    of the ONNX opset schema.
//
// We register the batchnorm shape inference functions via method (2) to ensure
// we have internal examples of this second code path.
static popart::RegisterShapeInferenceFunction
    batchnormRegister9(popart::Onnx::Operators::BatchNormalization_9,
                       batchnormShapeInferenceFun);

static popart::RegisterShapeInferenceFunction
    batchnormRegister7(popart::Onnx::Operators::BatchNormalization_7,
                       batchnormShapeInferenceFun);

static popart::RegisterShapeInferenceFunction
    batchnormRegister6(popart::Onnx::Operators::BatchNormalization_6,
                       batchnormShapeInferenceFun);

static popart::RegisterShapeInferenceFunction
    splitRegister1(popart::Onnx::Operators::Split_1, splitShapeInferenceFun);

static popart::RegisterShapeInferenceFunction
    splitRegister2(popart::Onnx::Operators::Split_2, splitShapeInferenceFun);

static popart::RegisterShapeInferenceFunction
    splitRegister11(popart::Onnx::Operators::Split_11, splitShapeInferenceFun);

static popart::RegisterShapeInferenceFunction
    tensorRemapRegister(popart::Onnx::CustomOperators::TensorRemap_1,
                        tensorRemapShapeInferenceFun);

} // namespace popart
