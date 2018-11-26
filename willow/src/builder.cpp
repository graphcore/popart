#include <poponnx/builder_impl.hpp>

namespace willow {

class TensorInfo;
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

TensorId Builder::abs(const std::vector<TensorId> &args) {
  return impl_->abs(args);
}

TensorId Builder::acos(const std::vector<TensorId> &args) {
  return impl_->acos(args);
}

TensorId Builder::acosh(const std::vector<TensorId> &args) {
  return impl_->acosh(args);
}

TensorId Builder::add(const std::vector<TensorId> &args) {
  return impl_->add(args);
}

TensorId Builder::logical_and(const std::vector<TensorId> &args) {
  return impl_->logical_and(args);
}

TensorId Builder::asin(const std::vector<TensorId> &args) {
  return impl_->asin(args);
}

TensorId Builder::asinh(const std::vector<TensorId> &args) {
  return impl_->asinh(args);
}

TensorId Builder::atan(const std::vector<TensorId> &args) {
  return impl_->atan(args);
}

TensorId Builder::atanh(const std::vector<TensorId> &args) {
  return impl_->atanh(args);
}

TensorId Builder::cast(const std::vector<TensorId> &args) {
  return impl_->cast(args);
}

TensorId Builder::ceil(const std::vector<TensorId> &args) {
  return impl_->ceil(args);
}

TensorId Builder::cos(const std::vector<TensorId> &args) {
  return impl_->cos(args);
}

TensorId Builder::cosh(const std::vector<TensorId> &args) {
  return impl_->cosh(args);
}

TensorId Builder::div(const std::vector<TensorId> &args) {
  return impl_->div(args);
}

TensorId Builder::elu(const std::vector<TensorId> &args) {
  return impl_->elu(args);
}

TensorId Builder::equal(const std::vector<TensorId> &args) {
  return impl_->equal(args);
}

TensorId Builder::exp(const std::vector<TensorId> &args) {
  return impl_->exp(args);
}

TensorId Builder::floor(const std::vector<TensorId> &args) {
  return impl_->floor(args);
}

TensorId Builder::greater(const std::vector<TensorId> &args) {
  return impl_->greater(args);
}

TensorId Builder::identity(const std::vector<TensorId> &args) {
  return impl_->identity(args);
}

TensorId Builder::less(const std::vector<TensorId> &args) {
  return impl_->less(args);
}

TensorId Builder::log(const std::vector<TensorId> &args) {
  return impl_->log(args);
}

TensorId Builder::max(const std::vector<TensorId> &args) {
  return impl_->max(args);
}

TensorId Builder::mean(const std::vector<TensorId> &args) {
  return impl_->mean(args);
}

TensorId Builder::min(const std::vector<TensorId> &args) {
  return impl_->min(args);
}

TensorId Builder::mul(const std::vector<TensorId> &args) {
  return impl_->mul(args);
}

TensorId Builder::neg(const std::vector<TensorId> &args) {
  return impl_->neg(args);
}

TensorId Builder::logical_not(const std::vector<TensorId> &args) {
  return impl_->logical_not(args);
}

TensorId Builder::logical_or(const std::vector<TensorId> &args) {
  return impl_->logical_or(args);
}

TensorId Builder::pow(const std::vector<TensorId> &args) {
  return impl_->pow(args);
}

TensorId Builder::reciprocal(const std::vector<TensorId> &args) {
  return impl_->reciprocal(args);
}

TensorId Builder::relu(const std::vector<TensorId> &args) {
  return impl_->relu(args);
}

TensorId Builder::sigmoid(const std::vector<TensorId> &args) {
  return impl_->sigmoid(args);
}

TensorId Builder::sin(const std::vector<TensorId> &args) {
  return impl_->sin(args);
}

TensorId Builder::sinh(const std::vector<TensorId> &args) {
  return impl_->sinh(args);
}

TensorId Builder::softsign(const std::vector<TensorId> &args) {
  return impl_->softsign(args);
}

TensorId Builder::sqrt(const std::vector<TensorId> &args) {
  return impl_->sqrt(args);
}

TensorId Builder::sub(const std::vector<TensorId> &args) {
  return impl_->sub(args);
}

TensorId Builder::sum(const std::vector<TensorId> &args) {
  return impl_->sum(args);
}

TensorId Builder::tan(const std::vector<TensorId> &args) {
  return impl_->tan(args);
}

TensorId Builder::tanh(const std::vector<TensorId> &args) {
  return impl_->tanh(args);
}

TensorId Builder::logical_xor(const std::vector<TensorId> &args) {
  return impl_->logical_xor(args);
}

TensorId Builder::convolution(const std::vector<TensorId> &args,
                              const std::vector<int64_t> strides,
                              const std::vector<int64_t> padding,
                              const std::vector<int64_t> dilation,
                              int64_t groups,
                              bool cacheOperation) {
  return impl_->convolution(
      args, strides, padding, dilation, groups, cacheOperation);
}

TensorId Builder::averagepool(const std::vector<TensorId> &args,
                              const std::vector<int64_t> kernel_shape,
                              const std::vector<int64_t> strides,
                              const std::vector<int64_t> padding) {
  return impl_->averagepool(args, kernel_shape, strides, padding);
}

TensorId Builder::maxpool(const std::vector<TensorId> &args,
                          const std::vector<int64_t> kernel_shape,
                          const std::vector<int64_t> strides,
                          const std::vector<int64_t> padding) {
  return impl_->maxpool(args, kernel_shape, strides, padding);
}

TensorId Builder::gemm(const std::vector<TensorId> &args,
                       float alpha,
                       float beta,
                       int64_t transA,
                       int64_t transB) {
  return impl_->gemm(args, alpha, beta, transA, transB);
}

TensorId Builder::pad(const std::vector<TensorId> &args,
                      std::string mode,
                      const std::vector<int64_t> pads,
                      float value) {
  return impl_->pad(args, mode, pads, value);
}

TensorId Builder::matmul(const std::vector<TensorId> &args) {
  return impl_->matmul(args);
}

TensorId Builder::softmax(const std::vector<TensorId> &args) {
  return impl_->softmax(args);
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
                               const std::vector<std::string> &attributeValue,
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

const std::map<std::string, TensorId> Builder::getTensorTranslation() const {
  return impl_->getTensorTranslation();
}

std::string Builder::getModelProto() const { return impl_->getModelProto(); }

std::vector<TensorId> Builder::getInputTensorIds() const {
  return impl_->getInputTensorIds();
}

std::vector<TensorId> Builder::getOutputTensorIds() const {
  return impl_->getOutputTensorIds();
}

std::vector<int64_t> Builder::getTensorShape(const TensorId id) {
  return impl_->getTensorShape(id);
}

} // namespace willow
