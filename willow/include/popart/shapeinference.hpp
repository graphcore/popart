// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_SHAPEINFERENCE_HPP
#define GUARD_SHAPEINFERENCE_HPP
#include <popart/attributes.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

// This is used as the only argument to the builders shape inference functions.
class ShapeInferenceContext {
public:
  ShapeInferenceContext(const std::map<int, TensorInfo> &inputInfos_,
                        const Attributes &);
  const TensorInfo &inInfo(int index) const;
  const Shape &inShape(int index) const;
  DataType inType(int index) const;
  TensorInfo &outInfo(int index);
  const std::map<int, TensorInfo> &getOutputInfos();
  const Attributes &getAttributes() const;

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

} // namespace popart

#endif
