#include <willow/onnxutil.hpp>
#include <willow/tensordata.hpp>

namespace willow {

TensorData::TensorData(const onnx::TensorProto &tp) {
  ConstVoidData cv_data = onnxutil::getConstData(tp);
  data_.resize(cv_data.info.nbytes());
  std::memcpy(data_.data(), cv_data.data, cv_data.info.nbytes());
}

TensorData::TensorData(const TensorInfo &info, const void *from) {
  data_.resize(info.nbytes());
  std::memcpy(data_.data(), from, info.nbytes());
}

void *TensorData::data() { return data_.data(); }

} // namespace willow
