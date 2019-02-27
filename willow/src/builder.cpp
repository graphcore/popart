#include <poponnx/builder_impl.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/opidentifier.hpp>

namespace poponnx {

class TensorInfo;

static void verifyWindowParameters(std::unique_ptr<BuilderImpl> &impl,
                                   TensorId input,
                                   const std::vector<int64_t> strides,
                                   const std::vector<int64_t> padding,
                                   const std::vector<int64_t> dilation = {}) {
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
                strides.size(),
                num_spatial_dims);
  }
  if (dilation.size() != 0 && dilation.size() != num_spatial_dims) {
    throw error(
        "Length of dilations vector {} != number of spatial dimensions {}",
        strides.size(),
        num_spatial_dims);
  }
}

// Functions that are expected by the generated code when the verifyInput
// is set to true.

static void
verify_AiOnnxOpset6_Conv_1(std::unique_ptr<BuilderImpl> &impl,
                           std::vector<TensorId> inputs,
                           std::map<std::string, boost::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      boost::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]),
      boost::any_cast<const std::vector<int64_t> &>(attributes["dilations"]));
}

static void verify_AiOnnxOpset6_AveragePool_1(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, boost::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      boost::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]));
}

static void verify_AiOnnxOpset7_AveragePool_7(
    std::unique_ptr<BuilderImpl> &impl,
    std::vector<TensorId> inputs,
    std::map<std::string, boost::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      boost::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]));
}

static void
verify_AiOnnxOpset6_MaxPool_1(std::unique_ptr<BuilderImpl> &impl,
                              std::vector<TensorId> inputs,
                              std::map<std::string, boost::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      boost::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]));
}

static void
verify_AiOnnxOpset8_MaxPool_8(std::unique_ptr<BuilderImpl> &impl,
                              std::vector<TensorId> inputs,
                              std::map<std::string, boost::any> attributes) {
  verifyWindowParameters(
      impl,
      inputs[0],
      boost::any_cast<const std::vector<int64_t> &>(attributes["strides"]),
      boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]));
}
static void
verify_AiOnnxOpset6_Pad_2(std::unique_ptr<BuilderImpl> &impl,
                          std::vector<TensorId> inputs,
                          std::map<std::string, boost::any> attributes) {

  auto rank = impl->getTensorShape(inputs[0]).size();
  auto &pads =
      boost::any_cast<const std::vector<int64_t> &>(attributes["pads"]);
  if (pads.size() != rank * 2) {
    throw error(
        "Padding vector (length {}) doesn't contain 2 entries per input "
        "dimension {}",
        pads.size(),
        rank);
  }
}

#include "builder.cpp.gen"

Builder::Builder() : impl_(new BuilderImpl()) {}

void Builder::configure() { impl_->configure(); }

std::unique_ptr<Builder> Builder::create() {
  auto builder = std::unique_ptr<Builder>(new Builder());
  builder->configure();
  return builder;
}

std::unique_ptr<Builder>
Builder::createFromOnnxModel(const std::string &modelProtoOrFilename) {
  auto builder = create();
  builder->loadModelProto(modelProtoOrFilename);
  return builder;
}

Builder::~Builder() {}

TensorId Builder::addInputTensor(const TensorInfo &tensorInfo) {
  return impl_->addInputTensor(tensorInfo);
}

TensorId Builder::addInitializedInputTensor(const ConstVoidData &initData) {
  return impl_->addInitializedInputTensor(initData);
}

void Builder::addOutputTensor(const TensorId &arg0) {
  return impl_->addOutputTensor(arg0);
}

std::vector<TensorId>
AiGraphcoreOpset1::groupnormalization(const std::vector<TensorId> &args,
                                      int64_t num_groups,
                                      float epsilon,
                                      const std::string &name) {
  std::map<std::string, boost::any> attributes;

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

std::vector<TensorId>
Builder::customOp(const OperatorIdentifier &opid,
                  int opsetVersion,
                  const std::vector<TensorId> &inputs,
                  const unsigned numOutputs,
                  const std::map<std::string, boost::any> &attributes,
                  const std::string &name) {
  return impl_->op(opid, opsetVersion, inputs, numOutputs, attributes, name);
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

void Builder::setAttribute(const std::string &attribute, boost::any value) {
  impl_->setAttribute(attribute, value);
}

void Builder::clearAttribute(const std::string &attribute) {
  impl_->clearAttribute(attribute);
}

void Builder::pushNameScope(const std::string &name) {
  impl_->pushNameScope(name);
}

void Builder::popNameScope() { impl_->popNameScope(); }

} // namespace poponnx
