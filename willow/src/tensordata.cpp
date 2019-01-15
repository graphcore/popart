#include <poponnx/error.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/tensordata.hpp>

#include <cstring>

namespace poponnx {

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

void TensorData::resetData(const onnx::TensorProto &tp) {
  ConstVoidData cv_data = onnxutil::getConstData(tp);
  if (data_.size() != cv_data.info.nbytes()) {
    throw error("can not reset tensor data with data of non-matching size");
  }
  std::memcpy(data_.data(), cv_data.data, cv_data.info.nbytes());
}

void TensorData::resetData(const TensorInfo &info, const void *from) {
  if (data_.size() != info.nbytes()) {
    throw error("can not reset tensor data with data of non-matching size");
  }
  data_.resize(info.nbytes());
  std::memcpy(data_.data(), from, info.nbytes());
}

template <> DataType ArrayWrapper<float>::getDtype() { return DataType::FLOAT; }
template <> DataType ArrayWrapper<int>::getDtype() { return DataType::INT32; }

template <>
std::ostream &operator<<(std::ostream &os, const ArrayWrapper<float> &array) {
  os << "{";
  for (int i = 0; i < array.getShape(0); ++i) {
    if (i != 0) {
      os << ", ";
    }

    os << (static_cast<float *>(array.data))[i];
  }
  os << "}";
  return os;
}

template <>
std::ostream &operator<<(std::ostream &os, const ArrayWrapper<int> &array) {

  os << "{";
  for (int n = 0; n < array.numElements(); ++n) {
    if (n != 0) {
      os << ", ";
    }
    os << array.data[n];
  }
  os << "}";
  return os;
}

} // namespace poponnx
