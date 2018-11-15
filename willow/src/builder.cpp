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

std::string Builder::add(const std::string &arg0, const std::string &arg1) {
  return impl_->add(arg0, arg1);
}

std::string Builder::convolution(const std::string &arg0,
                                 const std::string &arg1,
                                 const std::vector<int> strides,
                                 const std::vector<int> padding,
                                 const std::vector<int> dilation,
                                 int groups) {
  return impl_->convolution(arg0, arg1, strides, padding, dilation, groups);
}

std::string Builder::convolutionWithBias(const std::string &arg0,
                                         const std::string &arg1,
                                         const std::string &arg2,
                                         const std::vector<int> strides,
                                         const std::vector<int> padding,
                                         const std::vector<int> dilation,
                                         int groups) {
  return impl_->convolutionWithBias(
      arg0, arg1, arg2, strides, padding, dilation, groups);
}

std::string Builder::gemm(const std::string &arg0,
                          const std::string &arg1,
                          const std::string &arg2,
                          float alpha,
                          float beta,
                          int transA,
                          int transB) {
  return impl_->gemm(arg0, arg1, arg2, alpha, beta, transA, transB);
}

std::string Builder::matmul(const std::string &arg0, const std::string &arg1) {
  return impl_->matmul(arg0, arg1);
}

std::string Builder::getModelProto() const { return impl_->getModelProto(); }

} // namespace willow
