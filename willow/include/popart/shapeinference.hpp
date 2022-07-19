// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SHAPEINFERENCE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SHAPEINFERENCE_HPP_
#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <popart/attributes.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

// This is used as the only argument to the builders shape inference functions.
class ShapeInferenceContext {
public:
  ShapeInferenceContext(const std::map<int, TensorInfo> &inputInfos_,
                        const Attributes &,
                        int outputSize);
  const TensorInfo &inInfo(int index) const;
  const Shape &inShape(int index) const;
  const Shape &outShape(int index) const;
  DataType inType(int index) const;
  TensorInfo &outInfo(int index);
  const std::map<int, TensorInfo> &getOutputInfos();
  const Attributes &getAttributes() const;
  int getNumInputs() const;
  int getNumOutputs() const;
  /**
   * Check if tensor at input index n has an input shape.
   * Semantically equivalent to the implementation found in the ONNX package
   *
   * \param n The input index to check
   * \return true if tensor at input index n has a shape
   */
  bool hasInputShape(size_t n) const;
  /**
   * Check if all the tensors in the range [0, n) has an input shape.
   * Semantically equivalent to the implementation found in the ONNX package
   *
   * \param n The input index up to (but not including) to check
   * \return true if all the tensors has a shape
   */
  bool hasNInputShapes(size_t n) const;

  /**
   * Check if the proto contains an attribute
   *
   * \param key The attribute to check for
   * \return True if the proto contains an attribute
   */
  bool hasAttribute(const std::string &key) const;

  template <typename T> T getAttribute(const std::string &key) {
    return attributes.getAttribute<T>(key);
  }

  template <typename T>
  T getAttribute(const std::string &key, const T &defaultValue) {
    return attributes.getAttribute<T>(key, defaultValue);
  }

private:
  std::map<int, TensorInfo> inputInfos;
  std::map<int, TensorInfo> outputInfos;
  Attributes attributes;
  int outputSize;
};

using ShapeInferenceFunction = std::function<void(ShapeInferenceContext &)>;

// A singleton class for registering and accessing shape inference functions.
class ShapeInferenceFunctions {
public:
  static void registerFunction(const OperatorIdentifier &,
                               ShapeInferenceFunction);

  static bool hasFunction(const OperatorIdentifier &);
  static ShapeInferenceFunction getFunction(const OperatorIdentifier &);

private:
  // Make the constructor private.
  ShapeInferenceFunctions() {}
  static ShapeInferenceFunctions &getInstance();

  std::map<OperatorIdentifier, ShapeInferenceFunction> functions;
};

// Helper class used for static initialization of shape inference functions.
class RegisterShapeInferenceFunction {
public:
  RegisterShapeInferenceFunction(const OperatorIdentifier &opid,
                                 ShapeInferenceFunction);
};

// Popart equivalents of onnx functions in /onnx/defs/shape_inference.h.
inline void propagateElemTypeFromInputToOutput(ShapeInferenceContext &ctx,
                                               size_t inputIndex,
                                               size_t outputIndex) {
  ctx.outInfo(outputIndex) = {ctx.inType(inputIndex),
                              ctx.outInfo(outputIndex).shape()};
}

inline void propagateShapeFromInputToOutput(ShapeInferenceContext &ctx,
                                            size_t inputIndex,
                                            size_t outputIndex) {
  ctx.outInfo(outputIndex) = {ctx.outInfo(outputIndex).dataType(),
                              ctx.inShape(inputIndex)};
}

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_SHAPEINFERENCE_HPP_
