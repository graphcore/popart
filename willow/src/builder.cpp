// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
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

int64_t Builder::getPingPongPhase() const {
  if (!impl_->hasAttribute(sPingPongPhaseAttribute)) {
    throw popart::error("PingPong phase not set in current scope.");
  }
  return popart::any_cast<int64_t>(getAttribute(sPingPongPhaseAttribute));
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
                                   const std::vector<int64_t> strides,
                                   const std::vector<int64_t> padding,
                                   const std::vector<int64_t> kernel_shape,
                                   std::vector<int64_t> dilation = {}) {
  auto num_spatial_dims = impl->getTensorShape(input).size() - 2;
  if (num_spatial_dims < 1) {
    throw error("Input tensor has no spatial dimensions");
  }
  if (strides.size() != num_spatial_dims) {
    throw error(
        "Length of strides vector {} != number of spatial dimensions {}",
        strides.size(),
        num_spatial_dims);
  }
  if (padding.size() != num_spatial_dims * 2) {
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
  if (impl->hasTensorShape(input) && kernel_shape.size()) {
    // TODO T17932 : We do not have a mechanism for infering the output shape
    // of custom ops, so this check can only be applied if the tensor shape
    // is known
    Shape inShape = impl->getTensorShape(input);
    inShape.erase(inShape.begin(), inShape.begin() + 2);

    // Default 'ones'
    if (dilation.empty()) {
      dilation.resize(num_spatial_dims, 1);
    }

    Shape spatialOutShape = HasReceptiveFieldOp::getSpatialOutShape(
        inShape, kernel_shape, padding, strides, dilation);

    if (std::any_of(spatialOutShape.begin(),
                    spatialOutShape.end(),
                    [](int64_t i) { return i < 0; })) {
      throw error("Window parameter values combine to give invalid spatial "
                  "output shape: {}",
                  spatialOutShape);
    }
  }
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
  if (impl->hasTensorShape(inputs[1])) {
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

  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      weightsKShape,
      popart::any_cast<std::vector<int64_t> &>(attributes["dilations"]));
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
  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(
          attributes["kernel_shape"]));
}

static void verify_AiOnnxOpset7_AveragePool_7(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, popart::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(
          attributes["kernel_shape"]));
}

static void verify_AiOnnxOpset10_AveragePool_10(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, popart::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(
          attributes["kernel_shape"]));
}

static void verify_AiOnnxOpset11_AveragePool_11(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, popart::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(
          attributes["kernel_shape"]));
}

static void
verify_AiOnnxOpset6_MaxPool_1(std::unique_ptr<BuilderImpl> &impl,
                              std::vector<TensorId> inputs,
                              std::map<std::string, popart::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(
          attributes["kernel_shape"]));
}

static void
verify_AiOnnxOpset8_MaxPool_8(std::unique_ptr<BuilderImpl> &impl,
                              std::vector<TensorId> inputs,
                              std::map<std::string, popart::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(
          attributes["kernel_shape"]));
}

static void
verify_AiOnnxOpset10_MaxPool_10(std::unique_ptr<BuilderImpl> &impl,
                                std::vector<TensorId> inputs,
                                std::map<std::string, popart::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(
          attributes["kernel_shape"]));
}

static void
verify_AiOnnxOpset11_MaxPool_11(std::unique_ptr<BuilderImpl> &impl,
                                std::vector<TensorId> inputs,
                                std::map<std::string, popart::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      popart::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      popart::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      popart::any_cast<const std::vector<int64_t> &>(
          attributes["kernel_shape"]));
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

static void
verify_AiOnnxOpset11_Pad_11(std::unique_ptr<BuilderImpl> &impl,
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
                                        const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::PrintTensor,
           getOpsetVersion(),
           args,
           {{"print_gradient", print_gradient}},
           name)
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
                                   bool pass_through_creation,
                                   const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Detach,
           getOpsetVersion(),
           args,
           {{"pass_through_creation", pass_through_creation}},
           name)
      .at(0);
}

TensorId AiGraphcoreOpset1::init(Attributes::Ints shape,
                                 Attributes::Int data_type,
                                 Attributes::Int init_type,
                                 const std::string &name) {
  return impl
      ->op(Onnx::AiGraphcore::OpSet1::Init,
           getOpsetVersion(),
           {},
           {{"shape", shape},
            {"data_type", data_type},
            {"tensor_type", static_cast<int64_t>(TensorType::ActGrad)},
            {"init_type", init_type}},
           name)
      .at(0);
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
                                    const std::string &name) {
  std::string reductionString = LossOp::reductionTypeToString(reduction);

  std::map<std::string, popart::any> attributes = {
      {"reduction", reductionString}};
  if (ignoreIndex) {
    attributes.emplace("ignoreIndex", ignoreIndex);
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

std::vector<int64_t> Builder::getTensorShape(const TensorId id) {
  return impl_->getTensorShape(id);
}

std::string Builder::getTensorDtypeString(const TensorId id) {
  return impl_->getTensorDtypeString(id);
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
  auto nodeProto = impl_->findNodeProtoByOutputNames({nodeOutputName});
  if (nodeProto.op_type() != "Conv") {
    throw error("Builder::setPartialsType should only be called on Conv");
  }

  addNodeAttribute(sPartialsTypeAttribute, partialsType, {nodeOutputName});
}

std::string Builder::getPartialsType(const TensorId &nodeOutputName) {
  if (impl_->nodeHasAttribute(sPartialsTypeAttribute, {nodeOutputName})) {
    return impl_->getStringNodeAttribute(sPartialsTypeAttribute,
                                         {nodeOutputName});
  } else {
    return "FLOAT";
  }
}

void Builder::setAvailableMemoryProportion(
    const TensorId &nodeOutputName,
    const float availableMemoryProportion) {
  auto nodeProto = impl_->findNodeProtoByOutputNames({nodeOutputName});
  if (!(nodeProto.op_type() == "Conv" || nodeProto.op_type() == "MatMul")) {
    throw error("Builder::setAvailableMemoryProportion should only be called "
                "on Conv or MatMul");
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
