#include <poponnx/builder_impl.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/opidentifier.hpp>

namespace poponnx {

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

TensorId Builder::abs(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->abs(args, name);
}

TensorId Builder::acos(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->acos(args, name);
}

TensorId Builder::acosh(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->acosh(args, name);
}

TensorId Builder::add(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->add(args, name);
}

TensorId Builder::logical_and(const std::vector<TensorId> &args,
                              const std::string &name) {
  return impl_->logical_and(args, name);
}

TensorId Builder::asin(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->asin(args, name);
}

TensorId Builder::asinh(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->asinh(args, name);
}

TensorId Builder::atan(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->atan(args, name);
}

TensorId Builder::atanh(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->atanh(args, name);
}

TensorId Builder::cast(const std::vector<TensorId> &args,
                       DataType to,
                       const std::string &name) {
  return impl_->cast(args, onnxutil::getTPDataType(to), name);
}

TensorId Builder::ceil(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->ceil(args, name);
}

TensorId Builder::cos(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->cos(args, name);
}

TensorId Builder::cosh(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->cosh(args, name);
}

TensorId Builder::div(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->div(args, name);
}

TensorId Builder::elu(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->elu(args, name);
}

TensorId Builder::equal(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->equal(args, name);
}

TensorId Builder::exp(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->exp(args, name);
}

TensorId Builder::floor(const std::vector<TensorId> &args,
                        const std::string &name) {
  return impl_->floor(args, name);
}

TensorId Builder::greater(const std::vector<TensorId> &args,
                          const std::string &name) {
  return impl_->greater(args, name);
}

TensorId Builder::identity(const std::vector<TensorId> &args,
                           const std::string &name) {
  return impl_->identity(args, name);
}

TensorId Builder::less(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->less(args, name);
}

TensorId Builder::log(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->log(args, name);
}

TensorId Builder::max(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->max(args, name);
}

TensorId Builder::mean(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->mean(args, name);
}

TensorId Builder::min(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->min(args, name);
}

TensorId Builder::mul(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->mul(args, name);
}

TensorId Builder::neg(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->neg(args, name);
}

TensorId Builder::logical_not(const std::vector<TensorId> &args,
                              const std::string &name) {
  return impl_->logical_not(args, name);
}

TensorId Builder::logical_or(const std::vector<TensorId> &args,
                             const std::string &name) {
  return impl_->logical_or(args, name);
}

TensorId Builder::pow(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->pow(args, name);
}

TensorId Builder::reciprocal(const std::vector<TensorId> &args,
                             const std::string &name) {
  return impl_->reciprocal(args, name);
}

TensorId Builder::relu(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->relu(args, name);
}

TensorId Builder::sigmoid(const std::vector<TensorId> &args,
                          const std::string &name) {
  return impl_->sigmoid(args, name);
}

TensorId Builder::sin(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->sin(args, name);
}

TensorId Builder::sinh(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->sinh(args, name);
}

TensorId Builder::softsign(const std::vector<TensorId> &args,
                           const std::string &name) {
  return impl_->softsign(args, name);
}

TensorId Builder::sqrt(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->sqrt(args, name);
}

TensorId Builder::sub(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->sub(args, name);
}

TensorId Builder::sum(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->sum(args, name);
}

TensorId Builder::tan(const std::vector<TensorId> &args,
                      const std::string &name) {
  return impl_->tan(args, name);
}

TensorId Builder::tanh(const std::vector<TensorId> &args,
                       const std::string &name) {
  return impl_->tanh(args, name);
}

TensorId Builder::logical_xor(const std::vector<TensorId> &args,
                              const std::string &name) {
  return impl_->logical_xor(args, name);
}

TensorId Builder::convolution(const std::vector<TensorId> &args,
                              const std::vector<int64_t> strides,
                              const std::vector<int64_t> padding,
                              const std::vector<int64_t> dilation,
                              int64_t groups,
                              bool cacheOperation,
                              const std::string &name) {
  return impl_->convolution(
      args, strides, padding, dilation, groups, cacheOperation, name);
}

TensorId Builder::averagepool(const std::vector<TensorId> &args,
                              const std::vector<int64_t> kernel_shape,
                              const std::vector<int64_t> strides,
                              const std::vector<int64_t> padding,
                              const std::string &name) {
  return impl_->averagepool(args, kernel_shape, strides, padding, name);
}

TensorId Builder::maxpool(const std::vector<TensorId> &args,
                          const std::vector<int64_t> kernel_shape,
                          const std::vector<int64_t> strides,
                          const std::vector<int64_t> padding,
                          const std::string &name) {
  return impl_->maxpool(args, kernel_shape, strides, padding, name);
}

TensorId Builder::gemm(const std::vector<TensorId> &args,
                       float alpha,
                       float beta,
                       int64_t transA,
                       int64_t transB,
                       const std::string &name) {
  return impl_->gemm(args, alpha, beta, transA, transB, name);
}

TensorId Builder::pad(const std::vector<TensorId> &args,
                      std::string mode,
                      const std::vector<int64_t> pads,
                      float value,
                      const std::string &name) {
  return impl_->pad(args, mode, pads, value, name);
}

TensorId Builder::matmul(const std::vector<TensorId> &args,
                         const std::string &name) {
  return impl_->matmul(args, name);
}

TensorId Builder::softmax(const std::vector<TensorId> &args,
                          const std::string &name) {
  return impl_->softmax(args, name);
}

TensorId Builder::subsample(const std::vector<TensorId> &args,
                            const std::vector<int64_t> &strides,
                            const std::string &name) {
  return impl_->subsample(args, strides, name);
}

Builder::BatchNormalizationTrainingOutputs
Builder::batchnormalizationTraining(const TensorId x,
                                    const TensorId scale,
                                    const TensorId b,
                                    const TensorId mean,
                                    const TensorId var,
                                    const float epsilon,
                                    const float momentum,
                                    const int spatial,
                                    const std::string &name) {
  return impl_->batchnormalizationTraining(
      x, scale, b, mean, var, epsilon, momentum, spatial, name);
}

TensorId Builder::batchnormalizationTesting(const TensorId x,
                                            const TensorId scale,
                                            const TensorId b,
                                            const TensorId mean,
                                            const TensorId var,
                                            const float epsilon,
                                            const float momentum,
                                            const int spatial,
                                            const std::string &name) {
  return impl_->batchnormalizationTesting(
      x, scale, b, mean, var, epsilon, momentum, spatial, name);
}

TensorId Builder::transpose(const std::vector<TensorId> &args,
                            const std::vector<int64_t> &perm,
                            const std::string &name) {
  return impl_->transpose(args, perm, name);
}

TensorId Builder::reshape(const std::vector<TensorId> &args,
                          const std::string &name) {
  return impl_->reshape(args, name);
}

TensorId Builder::reshape_const(const std::vector<TensorId> &args,
                                const std::vector<int64_t> &shape,
                                const std::string &name) {
  return impl_->reshape_const(args, shape, name);
}

std::vector<TensorId> Builder::customOp(
    const OperatorIdentifier &opid,
    const std::vector<boost::any> &inputs,
    const unsigned numOutputs,
    const std::vector<std::pair<std::string, boost::any>> &attributes,
    const std::string &name) {
  return impl_->customOp(opid, inputs, numOutputs, attributes, name);
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

std::vector<TensorId> Builder::getValueTensorIds() const {
  return impl_->getValueTensorIds();
}

std::vector<int64_t> Builder::getTensorShape(const TensorId id) {
  return impl_->getTensorShape(id);
}

} // namespace poponnx
