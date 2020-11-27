// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <onnx/checker.h>
#include <popart/builder_impl.hpp>
#include <popart/filereader.hpp>
#include <popart/logging.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/receptive.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensor.hpp>

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
                                   const std::vector<int64_t> kernel_shape,
                                   std::vector<int64_t> dilation = {},
                                   const std::string &auto_pad   = "NOTSET",
                                   bool ceil_mode                = false) {
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

    // Validate that the input shape, kernel shape, strides, padding, and
    // optional dilation combine to produce a valid output shape
    if (kernel_shape.size()) {
      Shape inShape = impl->getTensorShape(input);
      inShape.erase(inShape.begin(), inShape.begin() + 2);

      // Default 'zeros'
      if (padding.empty()) {
        padding.resize(2 * num_spatial_dims, 0);
      }
      // Default 'ones'
      if (dilation.empty()) {
        dilation.resize(num_spatial_dims, 1);
      }
      if (strides.empty()) {
        strides.resize(num_spatial_dims, 1);
      }

      Shape spatialOutShape = HasReceptiveFieldOp::getSpatialOutShape(
          inShape,
          kernel_shape,
          padding,
          strides,
          dilation,
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
  if (!attributes.count("dilations")) {
    attributes["dilations"] = emptyVec;
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
      popart::any_cast<std::vector<int64_t> &>(attributes["kernel_shape"]),
      popart::any_cast<std::vector<int64_t> &>(attributes["dilations"]),
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
  if (!attributes.count("dilations")) {
    attributes["dilations"] = emptyVec;
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
      weightsKShape,
      popart::any_cast<std::vector<int64_t> &>(attributes["dilations"]),
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

Builder::Builder() : impl_(new BuilderImpl()) {}

void Builder::configure() { impl_->configure(); }

std::unique_ptr<Builder> Builder::create() {
  auto builder = std::unique_ptr<Builder>(new Builder());
  builder->configure();
  return builder;
}

Builder &Builder::createSubgraphBuilder() {
  children[nChildren] = create();
  auto child          = children[nChildren].get();
  ++nChildren;

  // point this to child
  impl_->addChild(child->impl_.get());

  // point child to this
  child->impl_->setParent(this->impl_.get());

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
                                 const std::string &debugPrefix) {
  return impl_->addInputTensor(tensorInfo, debugPrefix);
}

TensorId Builder::addInputTensor(const std::string &dataType,
                                 const Shape &shape,
                                 const std::string &debugPrefix) {
  return impl_->addInputTensor(TensorInfo(dataType, shape), debugPrefix);
}

TensorId Builder::addUntypedInputTensor(const std::string &debugPrefix) {
  return impl_->addUntypedInputTensor(debugPrefix);
}

void Builder::addInputTensorFromParentGraph(const TensorId &tensorId) {
  impl_->addInputTensorFromParentGraph(tensorId);
}

TensorId Builder::addInitializedInputTensor(const ConstVoidData &initData,
                                            const std::string &debugPrefix) {
  return impl_->addInitializedInputTensor(initData, debugPrefix);
}

void Builder::addOutputTensor(const TensorId &arg0) {
  return impl_->addOutputTensor(arg0);
}

std::vector<TensorId>
Builder::checkpointOutput(const std::vector<TensorId> &nodeOutputNames) {
  return impl_->checkpointOutput(nodeOutputNames);
}

std::vector<TensorId>
AiGraphcoreOpset1::groupnormalization(const std::vector<TensorId> &args,
                                      int64_t num_groups,
                                      float epsilon,
                                      const std::string &name) {
  std::map<std::string, popart::any> attributes;

  if (std::abs(epsilon - 1e-05f) > std::numeric_limits<float>::epsilon()) {
    attributes["epsilon"] = epsilon;
  }

  attributes["num_groups"] = num_groups;

  return impl->op(Onnx::AiGraphcore::OpSet1::GroupNormalization,
                  getOpsetVersion(),
                  args,
                  attributes,
                  name);
}

std::vector<TensorId> AiGraphcoreOpset1::multiconv(
    const MultiConvInputs &tensors,
    const MultiConvDilations &dilations,
    const MultiConvPads &pads,
    const MultiConvStrides &strides,
    const std::vector<float> &availableMemoryProportions,
    const std::vector<std::string> &partialsTypes,
    const nonstd::optional<std::string> planType,
    const nonstd::optional<int> perConvReservedTiles,
    const nonstd::optional<float> cycleBackOff,
    const std::string &name) {

  // Some checks:

  // 1. A multiconv must comprise at least one conv
  const auto numConvs = tensors.size();
  if (numConvs < 1) {
    throw error("MultiConvOp '{}' has no input tensors. Provide at least one "
                "set of inputs",
                name);
  }

  // 2. Each conv must have at least two inputs where the third bias input is
  // optional.
  for (size_t i = 0; i < numConvs; i++) {
    auto numConvInputs = tensors[i].size();
    if (numConvInputs < 2) {
      throw error("Each convolution of MultiConvOp '{}' must have at least two "
                  "inputs - data and weights",
                  name);
    }
    if (numConvInputs > 3) {
      throw error("Each convolution of MultiConvOp '{}' can have at most three "
                  "inputs - data, weights, and bias",
                  name);
    }
  }

  // 3. The number of parameters must equal the number of inputs,
  //    unless empty, in which case they take default values
  auto checkParamSize = [&numConvs, &name](int64_t paramsSize,
                                           std::string param) {
    if (paramsSize != 0 && paramsSize != numConvs) {
      throw error("For MultiConvOp '{}', number of {} parameter sets ({}) "
                  "does not match the number of input sets ({})",
                  name,
                  param,
                  paramsSize,
                  numConvs);
    }
  };
  checkParamSize(dilations.size(), "dilations");
  checkParamSize(pads.size(), "pads");
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
    attributes["dilations"] = dilations.empty() ? emptyParams : dilations[i];
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
  ConvPads flatPads;
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
    if (!pads.empty()) {
      flatPads.insert(end(flatPads), begin(pads[i]), end(pads[i]));
    }
    if (!strides.empty()) {
      flatStrides.insert(end(flatStrides), begin(strides[i]), end(strides[i]));
    }
  }

  std::map<std::string, popart::any> finalAttributes;
  finalAttributes["strides"]   = flatStrides;
  finalAttributes["pads"]      = flatPads;
  finalAttributes["dilations"] = flatDilations;

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

  return impl->op(Onnx::AiGraphcore::OpSet1::MultiConv,
                  getOpsetVersion(),
                  flatTensors,
                  static_cast<unsigned>(numConvs), // number of outputs
                  finalAttributes,
                  name);
}

TensorId AiGraphcoreOpset1::subsample(const std::vector<TensorId> &args,
                                      const std::vector<int64_t> &strides,
                                      const std::string &name) {

  for (int i = 0; i < strides.size(); ++i) {
    if (strides[i] == 0)
      throw error("Strides invalid. 0 stride at index {}", i);
  }

  return impl->op(Onnx::AiGraphcore::OpSet1::Subsample,
                  getOpsetVersion(),
                  args,
                  {{"strides", strides}},
                  name)[0];
}

TensorId AiGraphcoreOpset1::printtensor(const std::vector<TensorId> &args,
                                        int64_t print_gradient,
                                        const std::string &name,
                                        const std::string &title) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::PrintTensor,
           getOpsetVersion(),
           args,
           {{"print_gradient", print_gradient}, {"title", title}},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::nop(const std::vector<TensorId> &args,
                                const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Nop, getOpsetVersion(), args, {}, name)
      .at(0);
}

TensorId AiGraphcoreOpset1::scale(const std::vector<TensorId> &args,
                                  float scale,
                                  const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Scale,
           getOpsetVersion(),
           args,
           {{"scale", scale}},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::scaledadd(const std::vector<TensorId> &args,
                                      float scale0,
                                      float scale1,
                                      const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::ScaledAdd,
           getOpsetVersion(),
           args,
           {{"scale0", scale0}, {"scale1", scale1}},
           name)
      .at(0);
}

std::vector<TensorId> AiGraphcoreOpset1::lstm(const std::vector<TensorId> &args,
                                              int64_t outputFullSequence,
                                              const std::string &name) {
  return impl->op(Onnx::AiGraphcore::OpSet1::LSTM,
                  getOpsetVersion(),
                  args,
                  {{"output_full_sequence", outputFullSequence}},
                  name);
}

TensorId AiGraphcoreOpset1::gelu(const std::vector<TensorId> &args,
                                 const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Gelu, getOpsetVersion(), args, {}, name)
      .at(0);
}

TensorId AiGraphcoreOpset1::detach(const std::vector<TensorId> &args,
                                   const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Detach, getOpsetVersion(), args, {}, name)
      .at(0);
}

TensorId AiGraphcoreOpset1::round(const std::vector<TensorId> &args,
                                  const std::string &name) {
  std::map<std::string, popart::any> attributes;
  return impl->op(Onnx::AiGraphcore::OpSet1::Round,
                  getOpsetVersion(),
                  args,
                  attributes,
                  name)[0];
}

TensorId AiGraphcoreOpset1::init(Attributes::Ints shape,
                                 Attributes::Int data_type,
                                 Attributes::Int init_type,
                                 Attributes::Int batch_axis,
                                 const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Init,
           getOpsetVersion(),
           {},
           {{"shape", shape},
            {"data_type", data_type},
            {"tensor_type", static_cast<int64_t>(TensorType::ActGrad)},
            {"init_type", init_type},
            {"batch_axis", batch_axis}},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::init(Attributes::Ints shape,
                                 Attributes::Int data_type,
                                 Attributes::Int init_type,
                                 const std::string &name) {
  return AiGraphcoreOpset1::init(shape, data_type, init_type, -1, name);
}

TensorId AiGraphcoreOpset1::dynamicslice(const std::vector<TensorId> &args,
                                         Attributes::Ints axes,
                                         Attributes::Ints sizes,
                                         Attributes::Int noOverlap,
                                         const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::DynamicSlice,
           getOpsetVersion(),
           args,
           {{"axes", axes}, {"sizes", sizes}, {"noOverlap", noOverlap}},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::dynamicupdate(const std::vector<TensorId> &args,
                                          Attributes::Ints axes,
                                          Attributes::Ints sizes,
                                          Attributes::Int noOverlap,
                                          const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::DynamicUpdate,
           getOpsetVersion(),
           args,
           {{"axes", axes}, {"sizes", sizes}, {"noOverlap", noOverlap}},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::dynamiczero(const std::vector<TensorId> &args,
                                        Attributes::Ints axes,
                                        Attributes::Ints sizes,
                                        const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::DynamicZero,
           getOpsetVersion(),
           args,
           {
               {"axes", axes},
               {"sizes", sizes},
           },
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::dynamicadd(const std::vector<TensorId> &args,
                                       Attributes::Ints axes,
                                       Attributes::Ints sizes,
                                       const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::DynamicAdd,
           getOpsetVersion(),
           args,
           {
               {"axes", axes},
               {"sizes", sizes},
           },
           name)
      .at(0);
}

std::vector<TensorId> AiGraphcoreOpset1::call(const std::vector<TensorId> &args,
                                              unsigned num_outputs,
                                              const Builder &callee,
                                              const std::string &name) {

  ONNX_NAMESPACE::ModelProto modelProto =
      io::getModelFromString(callee.getModelProto());
  // May as well check the subgraph whilst we are here.

  // ONNX_NAMESPACE::checker::check_model(modelProto);
  ONNX_NAMESPACE::GraphProto calleeProto = modelProto.graph();
  // Some checks:
  // A subgraph must have at least one input and output, and the
  // number of inputs and outputs must match that of the callee
  // subgraph
  auto checkInOuts = [&](int64_t callSize, int64_t sgSize, std::string dir) {
    if (sgSize == 0) {
      throw error("CallOp subgraph requires at least one {}.", dir);
    }
    if (callSize != sgSize) {
      throw error("For CallOp '{}', number of {}s ({}) does not match that of "
                  "the callee subgraph ({})",
                  name,
                  dir,
                  callSize,
                  sgSize);
    }
  };
  checkInOuts(args.size(), calleeProto.input_size(), "input");
  checkInOuts(
      static_cast<int>(num_outputs), calleeProto.output_size(), "output");

  return impl->op(Onnx::AiGraphcore::OpSet1::Call,
                  getOpsetVersion(),
                  args,
                  num_outputs,
                  {{"callee", calleeProto}},
                  name);
}

TensorId
AiGraphcoreOpset1::replicatedallreduce(const std::vector<TensorId> &args,
                                       const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::ReplicatedAllReduce,
           getOpsetVersion(),
           args,
           {},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::l1loss(const std::vector<TensorId> &args,
                                   const float lambda,
                                   const ReductionType reduction,
                                   const std::string &name) {
  std::string reductionString = LossOp::reductionTypeToString(reduction);
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::L1,
           getOpsetVersion(),
           args,
           {{"lambda", lambda}, {"reduction", reductionString}},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::nllloss(const std::vector<TensorId> &args,
                                    const ReductionType reduction,
                                    const nonstd::optional<int> ignoreIndex,
                                    bool inputIsLogProbability,
                                    const std::string &name) {
  std::string reductionString = LossOp::reductionTypeToString(reduction);

  std::map<std::string, popart::any> attributes = {
      {"reduction", reductionString},
      {"inputIsLogProbability", inputIsLogProbability}};
  if (ignoreIndex.has_value()) {
    attributes.emplace("ignoreIndex", ignoreIndex.value());
  }

  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Nll,
           getOpsetVersion(),
           args,
           attributes,
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::identityloss(const std::vector<TensorId> &args,
                                         const ReductionType reduction,
                                         const std::string &name) {
  std::string reductionString = LossOp::reductionTypeToString(reduction);
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::IdentityLoss,
           getOpsetVersion(),
           args,
           {{"reduction", reductionString}},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::shapeddropout(const std::vector<TensorId> &args,
                                          const std::vector<int64_t> &shape,
                                          float ratio,
                                          const std::string &name) {
  std::map<std::string, popart::any> attributes = {{"shape", shape},
                                                   {"ratio", ratio}};

  return impl
      ->op(Onnx::AiGraphcore::OpSet1::ShapedDropout,
           getOpsetVersion(),
           args,
           attributes,
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::atan2(const std::vector<TensorId> &args,
                                  const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Atan2, getOpsetVersion(), args, {}, name)
      .at(0);
}

TensorId AiGraphcoreOpset1::expm1(const std::vector<TensorId> &args,
                                  const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Expm1, getOpsetVersion(), args, {}, name)
      .at(0);
}

TensorId AiGraphcoreOpset1::log1p(const std::vector<TensorId> &args,
                                  const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Log1p, getOpsetVersion(), args, {}, name)
      .at(0);
}

std::vector<TensorId>
Builder::customOp(const OperatorIdentifier &opid,
                  int opsetVersion,
                  const std::vector<TensorId> &inputs,
                  const unsigned numOutputs,
                  const std::map<std::string, popart::any> &attributes,
                  const std::string &name) {
  return impl_->op(opid, opsetVersion, inputs, numOutputs, attributes, name);
}

void Builder::customOp(const OperatorIdentifier &opid,
                       int opsetVersion,
                       const std::vector<TensorId> &inputs,
                       const std::vector<TensorId> &outputs,
                       const std::map<std::string, popart::any> &attributes,
                       const std::string &name) {
  impl_->op(opid, opsetVersion, inputs, outputs, attributes, name);
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
  if (!(nodeProto.op_type() == "Conv" || nodeProto.op_type() == "MatMul")) {
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
