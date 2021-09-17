// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/shapeinference.hpp>

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

int ShapeInferenceContext::getNumOutputs() const { return outputSize; }

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

// Shape inference is the process of infering the shape of the outputs of
// a node in the ONNX graph. It is run each time a new node is added to the
// model. It is executed inside BuilderImpl::runShapeInference via one of two
// code paths, depending how the shape inference function is registered:
// 1. registered in an Onnx operator schema. We do this for all aiGraphcore
//    and aiOnnx operators except for Batchnorm. See willow/src/defs.cc for
//    examples.
// 2. registered with the popart ShapeInferenceContext. This may be perferable
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

} // namespace popart
