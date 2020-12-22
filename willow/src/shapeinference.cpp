// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/shapeinference.hpp>

namespace popart {

ShapeInferenceContext::ShapeInferenceContext(
    const std::map<int, TensorInfo> &inputInfos_,
    const Attributes &attributes_)
    : inputInfos(inputInfos_), attributes(attributes_) {}

const TensorInfo &ShapeInferenceContext::inInfo(int index) const {
  return inputInfos.at(index);
}

const Shape &ShapeInferenceContext::inShape(int index) const {
  return inputInfos.at(index).shape();
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

} // namespace popart
