// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <builder_impl.hpp>
#include <builderdebuginfo.hpp>
#include <filereader.hpp>
#include <onnx/checker.h>
#include <onnxutil.hpp>
#include <popart/logging.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/receptive.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensor.hpp>
#include <poparttracepoint.hpp>

namespace popart {

int64_t Builder::getPipelineStage() const {
  if (!impl_->hasAttribute(sPipelineStageAttribute)) {
    throw popart::error("Pipeline stage not set in current scope.");
  }
  return popart::any_cast<int64_t>(getAttribute(sPipelineStageAttribute));
}

int64_t Builder::getExecutionPhase() const {
  if (!impl_->hasAttribute(sExecutionPhaseAttribute)) {
    throw popart::error("Execution phase not set in current scope.");
  }
  return popart::any_cast<int64_t>(getAttribute(sExecutionPhaseAttribute));
}

int64_t Builder::getVirtualGraph() const {
  if (!impl_->hasAttribute(sVirtualGraphAttribute)) {
    throw popart::error("Virtual graph not set in current scope.");
  }
  return popart::any_cast<int64_t>(getAttribute(sVirtualGraphAttribute));
}

class TensorInfo;

static void verifyWindowParameters(std::unique_ptr<BuilderImpl> &impl,
                                   TensorId input,
                                   std::vector<int64_t> strides,
                                   std::vector<int64_t> padding,
                                   std::vector<int64_t> outPadding,
                                   const std::vector<int64_t> kernel_shape,
                                   std::vector<int64_t> dilation   = {},
                                   std::vector<int64_t> inDilation = {},
                                   const std::string &auto_pad     = "NOTSET",
                                   bool ceil_mode                  = false) {

  // TODO T17932 : We do not have a mechanism for infering the output shape
  // of custom ops, so this set of checks can only be applied if the tensor
  // shape is known
  if (impl->hasTensorShape(input)) {
    auto num_spatial_dims = impl->getTensorShape(input).size() - 2;
    if (num_spatial_dims < 1) {
      throw error("Input tensor has no spatial dimensions");
    }
    if (strides.size() != 0 && strides.size() != num_spatial_dims) {
      throw error(
          "Length of strides vector {} != number of spatial dimensions {}",
          strides.size(),
          num_spatial_dims);
    }
    if (padding.size() != 0 && padding.size() != num_spatial_dims * 2) {
      throw error("Padding vector (length {}) does not have 2 values for each "
                  "spatial dimension {}",
                  padding.size(),
                  num_spatial_dims);
    }
    if (dilation.size() != 0 && dilation.size() != num_spatial_dims) {
      throw error(
          "Length of dilations vector {} != number of spatial dimensions {}",
          dilation.size(),
          num_spatial_dims);
    }
    if (inDilation.size() != 0 && inDilation.size() != num_spatial_dims) {
      throw error(
          "Length of inDilations vector {} != number of spatial dimensions {}",
          inDilation.size(),
          num_spatial_dims);
    }

    // Validate that the input shape, kernel shape, strides, padding, and
    // optional dilation combine to produce a valid output shape
    if (kernel_shape.size()) {
      Shape inShape = impl->getTensorShape(input);
      inShape.erase(inShape.begin(), inShape.begin() + 2);

      // Default 'zeros'
      if (padding.empty()) {
        padding.resize(2 * num_spatial_dims, 0);
      }
      if (outPadding.empty()) {
        outPadding.resize(2 * num_spatial_dims, 0);
      }
      // Default 'ones'
      if (dilation.empty()) {
        dilation.resize(num_spatial_dims, 1);
      }
      if (inDilation.empty()) {
        inDilation.resize(num_spatial_dims, 1);
      }
      if (strides.empty()) {
        strides.resize(num_spatial_dims, 1);
      }

      Shape spatialOutShape = HasReceptiveFieldOp::getSpatialOutShape(
          inShape,
          kernel_shape,
          padding,
          outPadding,
          strides,
          dilation,
          inDilation,
          HasReceptiveFieldOp::getAutoPad(auto_pad),
          ceil_mode);

      if (std::any_of(spatialOutShape.begin(),
                      spatialOutShape.end(),
                      [](int64_t i) { return i < 0; })) {
        throw error("Window parameter values combine to give invalid spatial "
                    "output shape: {}",
                    spatialOutShape);
      }
    }
  }
}

static void verifyPoolBase(std::unique_ptr<BuilderImpl> &impl,
                           std::vector<TensorId> inputs,
                           std::map<std::string, popart::any> attributes) {
  // Prepare attributes for verifyWindowParameters:
  // If attributes are unspecified (i.e. they do not
  // exist in the 'attributes' map) then set as empty
  std::vector<int64_t> emptyVec;
  if (!attributes.count("strides")) {
    attributes["strides"] = emptyVec;
  }
  if (!attributes.count("pads")) {
    attributes["pads"] = emptyVec;
  }
  if (!attributes.count("outPads")) {
    attributes["outPads"] = emptyVec;
  }
  if (!attributes.count("dilations")) {
    attributes["dilations"] = emptyVec;
  }
  if (!attributes.count("inDilations")) {
    attributes["inDilations"] = emptyVec;
  }
  std::string emptyString;
  if (!attributes.count("auto_pad")) {
    attributes["auto_pad"] = emptyString;
  }
  bool ceil_mode = false;
  if (attributes.count("ceil_mode")) {
    ceil_mode = popart::any_cast<int64_t>(attributes["ceil_mode"]);
  }

  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["outPads"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["kernel_shape"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["dilations"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["inDilations"]),
      popart::any_cast<const std::string &>(attributes["auto_pad"]),
      ceil_mode);
}

// Functions that are expected by the generated code when the verifyInput
// is set to true.

static void verifyConvBase(std::unique_ptr<BuilderImpl> &impl,
                           std::vector<TensorId> inputs,
                           std::map<std::string, popart::any> attributes) {
  // TODO T17932 : We do not have a mechanism for infering the output shape
  // of custom ops, so this check can only be applied if the tensor shape
  // is known
  Shape weightsKShape;
  if ((inputs.size() > 1) && impl->hasTensorShape(inputs[1])) {
    if (inputs.size() < 2) {
      throw error("Conv requires at least two inputs: data, and weights. {} "
                  "inputs provided.",
                  inputs.size());
    }
    weightsKShape = impl->getTensorShape(inputs[1]);
    weightsKShape.erase(weightsKShape.begin(), weightsKShape.begin() + 2);

    // Verify that the optional kernel_shape attribute matches the inferred
    // shape from the weight tensor's shape
    if (attributes.count("kernel_shape")) {
      auto userKShape =
          popart::any_cast<const Shape &>(attributes["kernel_shape"]);

      if (userKShape != weightsKShape) {
        throw error(
            "kernel_shape, {}, does not match inferred shape from weight "
            "input '{}', {}",
            userKShape,
            inputs[1],
            weightsKShape);
      }
    }
  }

  // Prepare attributes for verifyWindowParameters:
  // If attributes are unspecified (i.e. they do not
  // exist in the 'attributes' map) then set as empty
  std::vector<int64_t> emptyVec;
  if (!attributes.count("strides")) {
    attributes["strides"] = emptyVec;
  }
  if (!attributes.count("pads")) {
    attributes["pads"] = emptyVec;
  }
  if (!attributes.count("outPads")) {
    attributes["outPads"] = emptyVec;
  }
  if (!attributes.count("dilations")) {
    attributes["dilations"] = emptyVec;
  }
  if (!attributes.count("inDilations")) {
    attributes["inDilations"] = emptyVec;
  }
  std::string emptyString;
  if (!attributes.count("auto_pad")) {
    attributes["auto_pad"] = emptyString;
  }

  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["outPads"]),
      weightsKShape,
      popart::any_cast<std::vector<int64_t> &>(attributes["dilations"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["inDilations"]),
      popart::any_cast<std::string>(attributes["auto_pad"]));
}

static void
verify_AiOnnxOpset6_Conv_1(std::unique_ptr<BuilderImpl> &impl,
                           std::vector<TensorId> inputs,
                           std::map<std::string, popart::any> attributes) {
  verifyConvBase(impl, inputs, attributes);
}

static void
verify_AiOnnxOpset11_Conv_11(std::unique_ptr<BuilderImpl> &impl,
                             std::vector<TensorId> inputs,
                             std::map<std::string, popart::any> attributes) {
  verifyConvBase(impl, inputs, attributes);
}

static void verify_AiOnnxOpset6_AveragePool_1(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, popart::any> attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

static void verify_AiOnnxOpset7_AveragePool_7(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, popart::any> attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

static void verify_AiOnnxOpset10_AveragePool_10(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, popart::any> attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

static void verify_AiOnnxOpset11_AveragePool_11(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, popart::any> attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

static void
verify_AiOnnxOpset6_MaxPool_1(std::unique_ptr<BuilderImpl> &impl,
                              std::vector<TensorId> inputs,
                              std::map<std::string, popart::any> attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

static void
verify_AiOnnxOpset8_MaxPool_8(std::unique_ptr<BuilderImpl> &impl,
                              std::vector<TensorId> inputs,
                              std::map<std::string, popart::any> attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

static void
verify_AiOnnxOpset10_MaxPool_10(std::unique_ptr<BuilderImpl> &impl,
                                std::vector<TensorId> inputs,
                                std::map<std::string, popart::any> attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

static void
verify_AiOnnxOpset11_MaxPool_11(std::unique_ptr<BuilderImpl> &impl,
                                std::vector<TensorId> inputs,
                                std::map<std::string, popart::any> attributes) {
  verifyPoolBase(impl, inputs, attributes);
}

static void verifyPadBase(std::unique_ptr<BuilderImpl> &impl,
                          std::vector<TensorId> inputs,
                          std::map<std::string, popart::any> attributes) {

  auto rank = impl->getTensorShape(inputs[0]).size();
  auto &pads =
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]);
  if (pads.size() != rank * 2) {
    throw error(
        "Padding vector (length {}) doesn't contain 2 entries per input "
        "dimension {}",
        pads.size(),
        rank);
  }
}

static void
verify_AiOnnxOpset6_Pad_2(std::unique_ptr<BuilderImpl> &impl,
                          std::vector<TensorId> inputs,
                          std::map<std::string, popart::any> attributes) {
  verifyPadBase(impl, inputs, attributes);
}

#include "builder.cpp.gen"

Builder::Builder() : impl_(new BuilderImpl()), parent(nullptr) {}

void Builder::configure() { impl_->configure(); }

std::unique_ptr<Builder> Builder::create() {
  auto builder = std::unique_ptr<Builder>(new Builder());
  builder->configure();
  return builder;
}

void Builder::setParent(Builder *parent) {
  this->parent = parent;
  this->impl_->setParent(parent->impl_.get());
}

Builder *Builder::getParent() const { return parent; }

Builder &Builder::createSubgraphBuilder() {
  children[nChildren] = create();
  auto child          = children[nChildren].get();
  ++nChildren;

  // point this to child
  impl_->addChild(child->impl_.get());

  // Set the parent.
  child->setParent(this);

  return *child;
}

std::unique_ptr<Builder>
Builder::createFromOnnxModel(const std::string &modelProtoOrFilename) {
  auto builder = create();
  builder->loadModelProto(modelProtoOrFilename);
  return builder;
}

Builder::~Builder() {}

TensorId Builder::addInputTensor(const TensorInfo &tensorInfo,
                                 const popart::DebugContext &debugContext) {
  return impl_->addInputTensor(tensorInfo, debugContext);
}

TensorId Builder::addInputTensor(const std::string &dataType,
                                 const Shape &shape,
                                 const popart::DebugContext &debugContext) {
  return impl_->addInputTensor(TensorInfo(dataType, shape), debugContext);
}

TensorId Builder::addInputTensor(const TensorInfo &tensorInfo,
                                 const InputSettings &settings,
                                 const popart::DebugContext &debugContext) {
  return impl_->addInputTensor(tensorInfo, settings, debugContext);
}

TensorId Builder::addInputTensor(const std::string &dataType,
                                 const Shape &shape,
                                 const InputSettings &settings,
                                 const popart::DebugContext &debugContext) {
  return impl_->addInputTensor(
      TensorInfo(dataType, shape), settings, debugContext);
}

TensorId
Builder::addUntypedInputTensor(const popart::DebugContext &debugContext) {
  return impl_->addUntypedInputTensor(debugContext);
}

void Builder::addInputTensorFromParentGraph(const TensorId &tensorId) {
  impl_->addInputTensorFromParentGraph(tensorId);
}

TensorId
Builder::addInitializedInputTensor(const ConstVoidData &initData,
                                   const popart::DebugContext &debugContext) {
  return impl_->addInitializedInputTensor(initData, debugContext);
}

void Builder::addOutputTensor(const TensorId &arg0) {
  return impl_->addOutputTensor(arg0);
}

std::vector<TensorId>
Builder::checkpointOutput(const std::vector<TensorId> &nodeOutputNames) {
  return impl_->checkpointOutput(nodeOutputNames);
}

TensorId AiGraphcoreOpset1::copyvarupdate(const std::vector<TensorId> &args,
                                          const DebugContext &debugContext) {
  if (args.size() != 2) {
    throw error("copyvarupdate should have two args.");
  }

  // Copy var update itself does not need to have a variable input and is used
  // internally for non-variables. But for the sake of the builder API, we
  // can assume that a non-variable input is an error until a use-case is
  // found.
  if (!impl->isInitializer(args[0])) {
    throw error("The first arg of copyvarupdate should be an initalised "
                "tensor.");
  }

  std::map<std::string, popart::any> attributes;

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::CopyVarUpdate,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

std::vector<TensorId>
AiGraphcoreOpset1::groupnormalization(const std::vector<TensorId> &args,
                                      int64_t num_groups,
                                      float epsilon,
                                      const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;

  if (std::abs(epsilon - 1e-05f) > std::numeric_limits<float>::epsilon()) {
    attributes["epsilon"] = epsilon;
  }

  attributes["num_groups"] = num_groups;

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::GroupNormalization,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs;
}

std::vector<TensorId> AiGraphcoreOpset1::multiconv(
    const MultiConvInputs &tensors,
    const MultiConvDilations &dilations,
    const MultiConvDilations &inDilations,
    const MultiConvPads &pads,
    const MultiConvPads &outPads,
    const MultiConvStrides &strides,
    const std::vector<float> &availableMemoryProportions,
    const std::vector<std::string> &partialsTypes,
    const nonstd::optional<std::string> planType,
    const nonstd::optional<int> perConvReservedTiles,
    const nonstd::optional<float> cycleBackOff,
    const nonstd::optional<bool> enableConvDithering,
    const DebugContext &debugContext) {

  // Some checks:

  // 1. A multiconv must comprise at least one conv
  const auto numConvs = tensors.size();
  if (numConvs < 1) {
    throw error("MultiConvOp '{}' has no input tensors. Provide at least one "
                "set of inputs",
                debugContext.getPathName());
  }

  // 2. Each conv must have at least two inputs where the third bias input is
  // optional.
  for (size_t i = 0; i < numConvs; i++) {
    auto numConvInputs = tensors[i].size();
    if (numConvInputs < 2) {
      throw error("Each convolution of MultiConvOp '{}' must have at least two "
                  "inputs - data and weights",
                  debugContext.getPathName());
    }
    if (numConvInputs > 3) {
      throw error("Each convolution of MultiConvOp '{}' can have at most three "
                  "inputs - data, weights, and bias",
                  debugContext.getPathName());
    }
  }

  // 3. The number of parameters must equal the number of inputs,
  //    unless empty, in which case they take default values
  auto checkParamSize = [&numConvs, &debugContext](int64_t paramsSize,
                                                   std::string param) {
    if (paramsSize != 0 && paramsSize != numConvs) {
      throw error("For MultiConvOp '{}', number of {} parameter sets ({}) "
                  "does not match the number of input sets ({})",
                  debugContext.getPathName(),
                  param,
                  paramsSize,
                  numConvs);
    }
  };
  checkParamSize(dilations.size(), "dilations");
  checkParamSize(inDilations.size(), "inDilations");
  checkParamSize(pads.size(), "pads");
  checkParamSize(outPads.size(), "outPads");
  checkParamSize(strides.size(), "strides");
  checkParamSize(availableMemoryProportions.size(),
                 "availableMemoryProportions");
  checkParamSize(partialsTypes.size(), "partialsTypes");

  // 4. Check the parameters of each conv individually
  std::vector<int64_t> emptyParams;
  for (size_t i = 0; i < numConvs; i++) {
    std::map<std::string, popart::any> attributes;
    attributes["strides"]   = strides.empty() ? emptyParams : strides[i];
    attributes["pads"]      = pads.empty() ? emptyParams : pads[i];
    attributes["outPads"]   = outPads.empty() ? emptyParams : outPads[i];
    attributes["dilations"] = dilations.empty() ? emptyParams : dilations[i];
    attributes["inDilations"] =
        inDilations.empty() ? emptyParams : inDilations[i];
    verifyConvBase(impl, tensors[i], attributes);
  }

  // The the 'receptive field op' parameters - dilations, pads and
  // strides - are passed in as 'vectors of vectors'.
  // But ONNX does not support 'vector of vectors' as attributes.
  // For this reason, we flatten the parameters here to a 1D vector,
  // which is how they are represented in the ONNX model, and then
  // unflatten again in MultiConvBaseOp::setup().
  //
  // Similarly, for 'tensors'
  // tensors: {{data0, w0}, {data1, w1}, {data2, w2}}
  // flatTensors: {data0, w0, data1, w1, data2, w2}
  ConvInputs flatTensors;
  ConvDilations flatDilations;
  ConvDilations flatInDilations;
  ConvPads flatPads;
  ConvPads flatOutPads;
  ConvStrides flatStrides;
  for (size_t i = 0; i < numConvs; i++) {
    flatTensors.insert(
        flatTensors.end(), tensors[i].cbegin(), tensors[i].cend());

    if (tensors[i].size() == 2) {
      // Unbaised conv - insert empty placeholder for the optional bias
      flatTensors.emplace_back("");
    }

    // Flatten if not empty
    if (!dilations.empty()) {
      flatDilations.insert(
          end(flatDilations), begin(dilations[i]), end(dilations[i]));
    }
    if (!inDilations.empty()) {
      flatInDilations.insert(
          end(flatInDilations), begin(inDilations[i]), end(inDilations[i]));
    }
    if (!pads.empty()) {
      flatPads.insert(end(flatPads), begin(pads[i]), end(pads[i]));
    }
    if (!outPads.empty()) {
      flatOutPads.insert(end(flatOutPads), begin(outPads[i]), end(outPads[i]));
    }
    if (!strides.empty()) {
      flatStrides.insert(end(flatStrides), begin(strides[i]), end(strides[i]));
    }
  }

  std::map<std::string, popart::any> finalAttributes;
  if (flatStrides.size()) {
    finalAttributes["strides"] = flatStrides;
  }
  if (flatPads.size()) {
    finalAttributes["pads"] = flatPads;
  }
  if (flatOutPads.size()) {
    finalAttributes["outPads"] = flatOutPads;
  }
  if (flatDilations.size()) {
    finalAttributes["dilations"] = flatDilations;
  }
  if (flatInDilations.size()) {
    finalAttributes["inDilations"] = flatInDilations;
  }
  if (planType) {
    finalAttributes["planType"] = *planType;
  }
  if (perConvReservedTiles) {
    finalAttributes["perConvReservedTiles"] = *perConvReservedTiles;
  }
  if (cycleBackOff) {
    finalAttributes["cycleBackOff"] = *cycleBackOff;
  }
  if (availableMemoryProportions.size()) {
    finalAttributes[sAvailMemAttribute] = availableMemoryProportions;
  }
  if (partialsTypes.size()) {
    finalAttributes[sPartialsTypeAttribute] = partialsTypes;
  }
  finalAttributes["numConvs"] = static_cast<int64_t>(numConvs);
  if (enableConvDithering) {
    bool value{*enableConvDithering};
    finalAttributes[sEnableConvDitheringAttribute] =
        std::string(value ? "true" : "false");
  }

  BuilderDebugInfo di(
      debugContext, __POPART_FUNCTION_NAME__, flatTensors, finalAttributes);
  finalAttributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::MultiConv,
                          getOpsetVersion(),
                          flatTensors,
                          static_cast<unsigned>(numConvs), // number of outputs
                          finalAttributes,
                          {di});

  di.setOutputs(outputs);
  return outputs;
}

TensorId AiGraphcoreOpset1::subsample(const std::vector<TensorId> &args,
                                      const std::vector<int64_t> &strides,
                                      const DebugContext &debugContext) {

  for (int i = 0; i < strides.size(); ++i) {
    if (strides[i] == 0)
      throw error("Strides invalid. 0 stride at index {}", i);
  }

  std::map<std::string, popart::any> attributes = {{"strides", strides}};
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Subsample,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs[0];
}

TensorId AiGraphcoreOpset1::printtensor(const std::vector<TensorId> &args,
                                        int64_t print_gradient,
                                        const DebugContext &debugContext,
                                        const std::string &title) {
  std::map<std::string, popart::any> attributes = {
      {"print_gradient", print_gradient}, {"title", title}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::PrintTensor,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::nop(const std::vector<TensorId> &args,
                                const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Nop,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::scale(const std::vector<TensorId> &args,
                                  float scale,
                                  const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {{"scale", scale}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Scale,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});
  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::scaledadd(const std::vector<TensorId> &args,
                                      float scale0,
                                      float scale1,
                                      const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {{"scale0", scale0},
                                                   {"scale1", scale1}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::ScaledAdd,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});
  di.setOutputs(outputs);
  return outputs.at(0);
}

std::vector<TensorId>
AiGraphcoreOpset1::lstm(const std::vector<TensorId> &args,
                        int64_t outputFullSequence,
                        const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {
      {"output_full_sequence", outputFullSequence}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::LSTM,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs;
}

TensorId AiGraphcoreOpset1::gelu(const std::vector<TensorId> &args,
                                 const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Gelu,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::detach(const std::vector<TensorId> &args,
                                   const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Detach,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::depthtospace(const std::vector<TensorId> &args,
                                         int64_t blocksize,
                                         const std::string &mode,
                                         const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  attributes["blocksize"] = blocksize;
  if (mode != "DCR") {
    attributes["mode"] = mode;
  }

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::DepthToSpace,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::round(const std::vector<TensorId> &args,
                                  const DebugContext &debugContext) {

  std::map<std::string, popart::any> attributes;

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Round,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::init(Attributes::Ints shape,
                                 Attributes::Int data_type,
                                 Attributes::Int init_type,
                                 Attributes::Int batch_axis,
                                 const DebugContext &debugContext) {

  std::map<std::string, popart::any> attributes = {
      {"shape", shape},
      {"data_type", data_type},
      {"tensor_type", static_cast<int64_t>(TensorType::ActGrad)},
      {"init_type", init_type},
      {"batch_axis", batch_axis}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, {}, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(
      Onnx::AiGraphcore::OpSet1::Init, getOpsetVersion(), {}, attributes, {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::init(Attributes::Ints shape,
                                 Attributes::Int data_type,
                                 Attributes::Int init_type,
                                 const DebugContext &debugContext) {
  return AiGraphcoreOpset1::init(shape, data_type, init_type, -1, debugContext);
}

TensorId AiGraphcoreOpset1::dynamicslice(const std::vector<TensorId> &args,
                                         Attributes::Ints axes,
                                         Attributes::Ints sizes,
                                         Attributes::Int noOverlap,
                                         const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {
      {"axes", axes}, {"sizes", sizes}, {"noOverlap", noOverlap}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::DynamicSlice,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::dynamicupdate(const std::vector<TensorId> &args,
                                          Attributes::Ints axes,
                                          Attributes::Ints sizes,
                                          Attributes::Int noOverlap,
                                          const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {
      {"axes", axes}, {"sizes", sizes}, {"noOverlap", noOverlap}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::DynamicUpdate,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::dynamiczero(const std::vector<TensorId> &args,
                                        Attributes::Ints axes,
                                        Attributes::Ints sizes,
                                        const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {
      {"axes", axes},
      {"sizes", sizes},
  };

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::DynamicZero,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::dynamicadd(const std::vector<TensorId> &args,
                                       Attributes::Ints axes,
                                       Attributes::Ints sizes,
                                       const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {
      {"axes", axes},
      {"sizes", sizes},
  };

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::DynamicAdd,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::sequenceslice(const std::vector<TensorId> &args,
                                          Attributes::Int zeroUnused,
                                          const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {{"zeroUnused", zeroUnused}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::SequenceSlice,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

std::vector<TensorId>
AiGraphcoreOpset1::call(const std::vector<TensorId> &args,
                        unsigned num_outputs,
                        const Builder &callee,
                        const DebugContext &debugContext) {

  ONNX_NAMESPACE::ModelProto modelProto =
      io::getModelFromString(callee.getModelProto());
  // May as well check the subgraph whilst we are here.

  // ONNX_NAMESPACE::checker::check_model(modelProto);
  ONNX_NAMESPACE::GraphProto calleeProto = modelProto.graph();
  // Some checks:
  // The number of inputs and outputs must match that of the callee
  // subgraph
  auto checkInOuts = [&](int64_t callSize, int64_t sgSize, std::string dir) {
    if (callSize != sgSize) {
      throw error("For CallOp '{}', number of {}s ({}) does not match that of "
                  "the callee subgraph ({})",
                  debugContext.getPathName(),
                  dir,
                  callSize,
                  sgSize);
    }
  };
  checkInOuts(args.size(), calleeProto.input_size(), "input");
  checkInOuts(
      static_cast<int>(num_outputs), calleeProto.output_size(), "output");

  std::map<std::string, popart::any> attributes = {{"num_outputs", num_outputs},
                                                   {"callee", calleeProto}};
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Call,
                          getOpsetVersion(),
                          args,
                          num_outputs,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs;
}

TensorId AiGraphcoreOpset1::replicatedallreduce(
    const std::vector<TensorId> &args,
    const nonstd::optional<std::vector<int64_t>> &commGroup,
    const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});
  if (commGroup) {
    attributes.insert({sCollectiveCommGroup, *commGroup});
  }
  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::ReplicatedAllReduce,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::l1loss(const std::vector<TensorId> &args,
                                   const float lambda,
                                   const ReductionType reduction,
                                   const DebugContext &debugContext) {
  std::string reductionString = LossOp::reductionTypeToString(reduction);
  std::map<std::string, popart::any> attributes = {
      {"lambda", lambda}, {"reduction", reductionString}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});
  auto outputs = impl->op(
      Onnx::AiGraphcore::OpSet1::L1, getOpsetVersion(), args, attributes, {di});
  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::nllloss(const std::vector<TensorId> &args,
                                    const ReductionType reduction,
                                    const nonstd::optional<int> ignoreIndex,
                                    bool inputIsLogProbability,
                                    const DebugContext &debugContext) {
  std::string reductionString = LossOp::reductionTypeToString(reduction);

  std::map<std::string, popart::any> attributes = {
      {"reduction", reductionString},
      {"inputIsLogProbability", inputIsLogProbability}};
  if (ignoreIndex.has_value()) {
    attributes.emplace("ignoreIndex", ignoreIndex.value());
  }

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});
  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Nll,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});
  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::identityloss(const std::vector<TensorId> &args,
                                         const ReductionType reduction,
                                         const DebugContext &debugContext) {
  std::string reductionString = LossOp::reductionTypeToString(reduction);
  std::map<std::string, popart::any> attributes = {
      {"reduction", reductionString}};
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});
  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::IdentityLoss,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});
  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::ctcloss(const std::vector<TensorId> &args,
                                    const ReductionType reduction,
                                    const unsigned blank,
                                    const std::string &outDataType,
                                    const DebugContext &debugContext) {
  // Call _ctcloss but only return the first output.
  auto outputs = _ctcloss(args, reduction, blank, outDataType, debugContext);
  return outputs.at(0);
}

std::vector<TensorId>
AiGraphcoreOpset1::_ctcloss(const std::vector<TensorId> &args,
                            const ReductionType reduction,
                            const unsigned blank,
                            const std::string &outDataType,
                            const DebugContext &debugContext) {
  std::string reductionString = LossOp::reductionTypeToString(reduction);

  DataType toDataType = dataTypeFromString(outDataType);

  std::map<std::string, popart::any> attributes = {
      {"reduction", reductionString},
      {"blank", blank},
      {"outDataType", static_cast<int>(onnxutil::getTPDataType(toDataType))}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});
  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Ctc,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});
  di.setOutputs(outputs);
  return outputs;
}

std::vector<TensorId>
AiGraphcoreOpset1::ctcbeamsearchdecoder(const std::vector<TensorId> &args,
                                        const unsigned blank,
                                        const unsigned beamWidth,
                                        const unsigned topPaths,
                                        const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {
      {"blank", blank},
      {"beam_width", beamWidth},
      {"top_paths", topPaths},
  };
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});
  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::CtcBeamSearchDecoder,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});
  di.setOutputs(outputs);
  return outputs;
}

TensorId AiGraphcoreOpset1::shapeddropout(const std::vector<TensorId> &args,
                                          const std::vector<int64_t> &shape,
                                          float ratio,
                                          const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {{"shape", shape},
                                                   {"ratio", ratio}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});
  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::ShapedDropout,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});
  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::atan2(const std::vector<TensorId> &args,
                                  const DebugContext &debugContext) {

  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});
  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Atan2,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});
  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::expm1(const std::vector<TensorId> &args,
                                  const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Expm1,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::log1p(const std::vector<TensorId> &args,
                                  const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Log1p,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}
TensorId AiGraphcoreOpset1::reshape(const TensorId &arg,
                                    const Attributes::Ints &shape,
                                    const DebugContext &debugContext) {

  std::map<std::string, popart::any> attributes = {{"shape", shape}};
  BuilderDebugInfo di(
      debugContext, __POPART_FUNCTION_NAME__, {arg}, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Reshape,
                          getOpsetVersion(),
                          {arg},
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::fmod(const std::vector<TensorId> &args,
                                 const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.emplace(sDebugInfoId, di.getId());

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Fmod,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::remainder(const std::vector<TensorId> &args,
                                      const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Remainder,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::reverse(const std::vector<TensorId> &args,
                                    const std::vector<int64_t> &dimensions,
                                    const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {{"dimensions", dimensions}};
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Reverse,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::packedDataBlock(
    const std::vector<TensorId> &args,
    const std::vector<int64_t> &maxSequenceLengths,
    int64_t resultSize,
    int64_t callbackBatchSize,
    const Builder &callback,
    const DebugContext &debugContext) {

  ONNX_NAMESPACE::ModelProto modelProto =
      io::getModelFromString(callback.getModelProto());
  ONNX_NAMESPACE::GraphProto callbackProto = modelProto.graph();

  std::map<std::string, popart::any> attributes = {
      {"resultSize", resultSize},
      {"callbackBatchSize", callbackBatchSize},
      {"maxSequenceLengths", maxSequenceLengths},
      {"callback", callbackProto}};

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::PackedDataBlock,
                          getOpsetVersion(),
                          args,
                          attributes,
                          debugContext);
  di.setOutputs(outputs);
  return outputs.at(0);
}

void AiGraphcoreOpset1::abort(const std::vector<TensorId> &args,
                              const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  impl->op(Onnx::AiGraphcore::OpSet1::Abort,
           getOpsetVersion(),
           args,
           attributes,
           {di});
}

TensorId AiGraphcoreOpset1::bitwiseGenericOp(const OperatorIdentifier &opid,
                                             const std::vector<TensorId> &args,
                                             const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(opid, getOpsetVersion(), args, attributes, {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::bitwisenot(const std::vector<TensorId> &args,
                                       const DebugContext &debugContext) {
  return bitwiseGenericOp(
      Onnx::AiGraphcore::OpSet1::BitwiseNot, args, debugContext);
}

TensorId AiGraphcoreOpset1::bitwiseand(const std::vector<TensorId> &args,
                                       const DebugContext &debugContext) {
  return bitwiseGenericOp(
      Onnx::AiGraphcore::OpSet1::BitwiseAnd, args, debugContext);
}

TensorId AiGraphcoreOpset1::bitwiseor(const std::vector<TensorId> &args,
                                      const DebugContext &debugContext) {
  return bitwiseGenericOp(
      Onnx::AiGraphcore::OpSet1::BitwiseOr, args, debugContext);
}

TensorId AiGraphcoreOpset1::bitwisexor(const std::vector<TensorId> &args,
                                       const DebugContext &debugContext) {
  return bitwiseGenericOp(
      Onnx::AiGraphcore::OpSet1::BitwiseXor, args, debugContext);
}

TensorId AiGraphcoreOpset1::bitwisexnor(const std::vector<TensorId> &args,
                                        const DebugContext &debugContext) {
  return bitwiseGenericOp(
      Onnx::AiGraphcore::OpSet1::BitwiseXnor, args, debugContext);
}

std::vector<TensorId> AiGraphcoreOpset1::reducemedian(
    const std::vector<TensorId> &args,
    const nonstd::optional<std::vector<int64_t>> &axes,
    int64_t keepdims,
    const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes = {{"keepdims", keepdims}};

  if (axes) {
    attributes["axes"] = *axes;
  }

  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::ReduceMedian,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs;
}

TensorId AiGraphcoreOpset1::scatterreduce(const std::vector<TensorId> &args,
                                          Attributes::Int axis_size,
                                          Attributes::Int axis,
                                          ScatterReduction reduction,
                                          const DebugContext &debugContext) {
  auto reductionStr = ScatterReduceOp::reductionToString(reduction);

  std::map<std::string, popart::any> attributes = {
      {"axis", axis}, {"axis_size", axis_size}, {"reduction", reductionStr}};
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::ScatterReduce,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

TensorId AiGraphcoreOpset1::swish(const std::vector<TensorId> &args,
                                  const DebugContext &debugContext) {
  std::map<std::string, popart::any> attributes;
  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);
  attributes.insert({sDebugInfoId, di.getId()});

  auto outputs = impl->op(Onnx::AiGraphcore::OpSet1::Swish,
                          getOpsetVersion(),
                          args,
                          attributes,
                          {di});

  di.setOutputs(outputs);
  return outputs.at(0);
}

std::vector<TensorId>
Builder::customOp(const OperatorIdentifier &opid,
                  int opsetVersion,
                  const std::vector<TensorId> &inputs,
                  const unsigned numOutputs,
                  const std::map<std::string, popart::any> &attributes,
                  const DebugContext &debugContext) {
  std::map<std::string, popart::any> _attributes = attributes;
  BuilderDebugInfo di(
      debugContext, __POPART_FUNCTION_NAME__, inputs, attributes);
  _attributes.insert({sDebugInfoId, di.getId()});

  auto outputs =
      impl_->op(opid, opsetVersion, inputs, numOutputs, _attributes, {di});

  di.setOutputs(outputs);
  return outputs;
}

void Builder::customOp(const OperatorIdentifier &opid,
                       int opsetVersion,
                       const std::vector<TensorId> &inputs,
                       const std::vector<TensorId> &outputs,
                       const std::map<std::string, popart::any> &attributes,
                       const DebugContext &debugContext) {
  std::map<std::string, popart::any> _attributes = attributes;
  BuilderDebugInfo di(
      debugContext, __POPART_FUNCTION_NAME__, inputs, attributes, outputs);
  _attributes.insert({sDebugInfoId, di.getId()});

  impl_->op(opid, opsetVersion, inputs, outputs, _attributes, {di});
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const int64_t &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const std::vector<int64_t> &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const float &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const std::vector<float> &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const std::string &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const char *attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const std::vector<std::string> &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const bool attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

void Builder::addNodeAttribute(const std::string &attributeName,
                               const ConstVoidData &attributeValue,
                               const std::set<TensorId> &nodeOutputNames) {
  impl_->addNodeAttribute(attributeName, attributeValue, nodeOutputNames);
}

bool Builder::nodeHasAttribute(const std::string &attributeName,
                               const std::set<TensorId> &nodeOutputNames) {
  return impl_->nodeHasAttribute(attributeName, nodeOutputNames);
}

int64_t
Builder::getInt64NodeAttribute(const std::string &attributeName,
                               const std::set<TensorId> &nodeOutputNames) {
  return impl_->getInt64NodeAttribute(attributeName, nodeOutputNames);
}

std::vector<int64_t> Builder::getInt64VectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  return impl_->getInt64VectorNodeAttribute(attributeName, nodeOutputNames);
}

float Builder::getFloatNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  return impl_->getFloatNodeAttribute(attributeName, nodeOutputNames);
}

std::vector<float> Builder::getFloatVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  return impl_->getFloatVectorNodeAttribute(attributeName, nodeOutputNames);
}

std::string
Builder::getStringNodeAttribute(const std::string &attributeName,
                                const std::set<TensorId> &nodeOutputNames) {
  return impl_->getStringNodeAttribute(attributeName, nodeOutputNames);
}

std::vector<std::string> Builder::getStringVectorNodeAttribute(
    const std::string &attributeName,
    const std::set<TensorId> &nodeOutputNames) {
  return impl_->getStringVectorNodeAttribute(attributeName, nodeOutputNames);
}

bool Builder::getBoolNodeAttribute(const std::string &attributeName,
                                   const std::set<TensorId> &nodeOutputNames) {
  return impl_->getBoolNodeAttribute(attributeName, nodeOutputNames);
}

void Builder::removeNodeAttribute(const std::string &attributeName,
                                  const std::set<TensorId> &nodeOutputNames) {
  impl_->removeNodeAttribute(attributeName, nodeOutputNames);
}

std::vector<std::string>
Builder::getAllNodeAttributeNames(const std::set<TensorId> &nodeOutputNames) {
  return impl_->getAllNodeAttributeNames(nodeOutputNames);
}

void Builder::loadModelProto(const std::string &modelProtoOrFilename) {
  impl_->loadModelProto(modelProtoOrFilename);
}

void Builder::saveModelProto(const std::string &fn) {
  impl_->saveModelProto(fn);
}

void Builder::saveInitializersExternally(const std::vector<TensorId> &ids,
                                         const std::string &fn) {
  impl_->saveInitializersExternally(ids, fn);
}

std::string Builder::getModelProto() const { return impl_->getModelProto(); }

std::vector<TensorId> Builder::getInputTensorIds() const {
  return impl_->getInputTensorIds();
}

std::vector<TensorId> Builder::getOutputTensorIds() const {
  return impl_->getOutputTensorIds();
}

std::vector<TensorId> Builder::getValueTensorIds() const {
  return impl_->getValueTensorIds();
}

std::vector<TensorId> Builder::getTrainableTensorIds() const {
  return impl_->getTrainableTensorIds();
}

bool Builder::hasValueInfo(const TensorId &id) const {
  return impl_->hasValueInfo(id);
}

std::vector<int64_t> Builder::getTensorShape(const TensorId id) {
  return impl_->getTensorShape(id);
}

std::string Builder::getTensorDtypeString(const TensorId id) {
  return impl_->getTensorDtypeString(id);
}

DataType Builder::getTensorDataType(const TensorId id) {
  return impl_->getTensorDataType(id);
}

bool Builder::isInitializer(const TensorId id) const {
  return impl_->isInitializer(id);
}

void Builder::setAttribute(const std::string &attribute, popart::any value) {
  impl_->setAttribute(attribute, value);
}

popart::any Builder::getAttribute(const std::string attribute) const {
  return impl_->getAttribute(attribute);
}

bool Builder::hasAttribute(const std::string &attribute) const {
  return impl_->hasAttribute(attribute);
}

void Builder::clearAttribute(const std::string &attribute) {
  impl_->clearAttribute(attribute);
}

bool Builder::hasAttribute(const std::string &attribute) {
  return impl_->hasAttribute(attribute);
}

popart::any Builder::getAttribute(const std::string &attribute) {
  return impl_->getAttribute(attribute);
}

void Builder::pushNameScope(const std::string &name) {
  impl_->pushNameScope(name);
}

std::string Builder::getNameScope(const std::string &name) const {
  return impl_->getNameScope(name);
}

void Builder::popNameScope() { impl_->popNameScope(); }

void Builder::setPartialsType(const TensorId &nodeOutputName,
                              const std::string partialsType) {
  std::vector<std::string> supportedOps{"Conv", "MatMul"};

  const auto nodeProto = impl_->findNodeProtoByOutputNames({nodeOutputName});
  const auto opType    = nodeProto.op_type();

  const bool opSupportsPartials =
      std::find(supportedOps.begin(), supportedOps.end(), opType) !=
      supportedOps.end();

  if (!opSupportsPartials) {
    throw error("Builder::setPartialsType should only be called on operators: "
                "Conv, MatMul; but was given: " +
                opType);
  }

  addNodeAttribute(sPartialsTypeAttribute, partialsType, {nodeOutputName});
}

std::string Builder::getPartialsType(const TensorId &nodeOutputName) {
  if (impl_->nodeHasAttribute(sPartialsTypeAttribute, {nodeOutputName})) {
    return impl_->getStringNodeAttribute(sPartialsTypeAttribute,
                                         {nodeOutputName});
  } else {
    // Poplar default partial type.
    return "FLOAT";
  }
}

void Builder::setAvailableMemoryProportion(
    const TensorId &nodeOutputName,
    const float availableMemoryProportion) {
  auto nodeProto = impl_->findNodeProtoByOutputNames({nodeOutputName});
  std::vector<std::string> supported_ops = {
      "Conv", "MatMul", "Gather", "ScatterReduce"};
  bool not_supported_op = std::find(supported_ops.begin(),
                                    supported_ops.end(),
                                    nodeProto.op_type()) == supported_ops.end();

  if (not_supported_op) {
    return;
  } else if (availableMemoryProportion > 1.0f ||
             availableMemoryProportion <= 0.0f) {
    throw error("availableMemoryProportion must be in (0,1]");
  }

  addNodeAttribute(
      sAvailMemAttribute, availableMemoryProportion, {nodeOutputName});
}

void Builder::setGraphName(const std::string &name) {
  return impl_->setGraphName(name);
}

} // namespace popart
