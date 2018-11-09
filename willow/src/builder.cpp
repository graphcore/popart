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

std::string Builder::getModelProto() const { return impl_->getModelProto(); }

} // namespace willow
