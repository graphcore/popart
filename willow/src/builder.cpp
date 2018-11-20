#include <poponnx/builder_impl.hpp>

namespace willow {

class TensorInfo;

Builder::Builder() : impl_(new BuilderImpl()) {}

Builder::~Builder() {}

std::string Builder::addInputTensor(const TensorInfo &tensorInfo) {
  return impl_->addInputTensor(tensorInfo);
}

void Builder::addOutputTensor(const std::string &arg0) {
  return impl_->addOutputTensor(arg0);
}

std::string Builder::abs(const std::vector<std::string> &args) {
  return impl_->abs(args);
}

std::string Builder::acos(const std::vector<std::string> &args) {
  return impl_->acos(args);
}

std::string Builder::acosh(const std::vector<std::string> &args) {
  return impl_->acosh(args);
}

std::string Builder::add(const std::vector<std::string> &args) {
  return impl_->add(args);
}

std::string Builder::logical_and(const std::vector<std::string> &args) {
  return impl_->logical_and(args);
}

std::string Builder::asin(const std::vector<std::string> &args) {
  return impl_->asin(args);
}

std::string Builder::asinh(const std::vector<std::string> &args) {
  return impl_->asinh(args);
}

std::string Builder::atan(const std::vector<std::string> &args) {
  return impl_->atan(args);
}

std::string Builder::atanh(const std::vector<std::string> &args) {
  return impl_->atanh(args);
}

std::string Builder::cast(const std::vector<std::string> &args) {
  return impl_->cast(args);
}

std::string Builder::ceil(const std::vector<std::string> &args) {
  return impl_->ceil(args);
}

std::string Builder::cos(const std::vector<std::string> &args) {
  return impl_->cos(args);
}

std::string Builder::cosh(const std::vector<std::string> &args) {
  return impl_->cosh(args);
}

std::string Builder::div(const std::vector<std::string> &args) {
  return impl_->div(args);
}

std::string Builder::elu(const std::vector<std::string> &args) {
  return impl_->elu(args);
}

std::string Builder::equal(const std::vector<std::string> &args) {
  return impl_->equal(args);
}

std::string Builder::exp(const std::vector<std::string> &args) {
  return impl_->exp(args);
}

std::string Builder::floor(const std::vector<std::string> &args) {
  return impl_->floor(args);
}

std::string Builder::greater(const std::vector<std::string> &args) {
  return impl_->greater(args);
}

std::string Builder::identity(const std::vector<std::string> &args) {
  return impl_->identity(args);
}

std::string Builder::less(const std::vector<std::string> &args) {
  return impl_->less(args);
}

std::string Builder::log(const std::vector<std::string> &args) {
  return impl_->log(args);
}

std::string Builder::max(const std::vector<std::string> &args) {
  return impl_->max(args);
}

std::string Builder::mean(const std::vector<std::string> &args) {
  return impl_->mean(args);
}

std::string Builder::min(const std::vector<std::string> &args) {
  return impl_->min(args);
}

std::string Builder::mul(const std::vector<std::string> &args) {
  return impl_->mul(args);
}

std::string Builder::neg(const std::vector<std::string> &args) {
  return impl_->neg(args);
}

std::string Builder::logical_not(const std::vector<std::string> &args) {
  return impl_->logical_not(args);
}

std::string Builder::logical_or(const std::vector<std::string> &args) {
  return impl_->logical_or(args);
}

std::string Builder::pow(const std::vector<std::string> &args) {
  return impl_->pow(args);
}

std::string Builder::reciprocal(const std::vector<std::string> &args) {
  return impl_->reciprocal(args);
}

std::string Builder::relu(const std::vector<std::string> &args) {
  return impl_->relu(args);
}

std::string Builder::sigmoid(const std::vector<std::string> &args) {
  return impl_->sigmoid(args);
}

std::string Builder::sin(const std::vector<std::string> &args) {
  return impl_->sin(args);
}

std::string Builder::sinh(const std::vector<std::string> &args) {
  return impl_->sinh(args);
}

std::string Builder::softsign(const std::vector<std::string> &args) {
  return impl_->softsign(args);
}

std::string Builder::sqrt(const std::vector<std::string> &args) {
  return impl_->sqrt(args);
}

std::string Builder::sub(const std::vector<std::string> &args) {
  return impl_->sub(args);
}

std::string Builder::sum(const std::vector<std::string> &args) {
  return impl_->sum(args);
}

std::string Builder::tan(const std::vector<std::string> &args) {
  return impl_->tan(args);
}

std::string Builder::tanh(const std::vector<std::string> &args) {
  return impl_->tanh(args);
}

std::string Builder::logical_xor(const std::vector<std::string> &args) {
  return impl_->logical_xor(args);
}

std::string Builder::convolution(const std::vector<std::string> &args,
                                 const std::vector<int> strides,
                                 const std::vector<int> padding,
                                 const std::vector<int> dilation,
                                 int groups) {
  return impl_->convolution(args, strides, padding, dilation, groups);
}

std::string Builder::gemm(const std::vector<std::string> &args,
                          float alpha,
                          float beta,
                          int transA,
                          int transB) {
  return impl_->gemm(args, alpha, beta, transA, transB);
}

std::string Builder::matmul(const std::vector<std::string> &args) {
  return impl_->matmul(args);
}

std::string Builder::getModelProto() const { return impl_->getModelProto(); }

} // namespace willow
