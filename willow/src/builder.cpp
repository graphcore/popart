#include <poponnx/builder_impl.hpp>

namespace willow {

class TensorInfo;

Builder::Builder() : impl_(new BuilderImpl()) {}

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
                              const std::vector<int> strides,
                              const std::vector<int> padding,
                              const std::vector<int> dilation,
                              int groups) {
  return impl_->convolution(args, strides, padding, dilation, groups);
}

TensorId Builder::averagepool(const std::vector<TensorId> &args,
                              const std::vector<int> kernel_shape,
                              const std::vector<int> strides,
                              const std::vector<int> padding) {
  return impl_->averagepool(args, kernel_shape, strides, padding);
}

TensorId Builder::maxpool(const std::vector<TensorId> &args,
                          const std::vector<int> kernel_shape,
                          const std::vector<int> strides,
                          const std::vector<int> padding) {
  return impl_->maxpool(args, kernel_shape, strides, padding);
}

TensorId Builder::gemm(const std::vector<TensorId> &args,
                       float alpha,
                       float beta,
                       int transA,
                       int transB) {
  return impl_->gemm(args, alpha, beta, transA, transB);
}

TensorId Builder::matmul(const std::vector<TensorId> &args) {
  return impl_->matmul(args);
}

std::string Builder::getModelProto() const { return impl_->getModelProto(); }

} // namespace willow
