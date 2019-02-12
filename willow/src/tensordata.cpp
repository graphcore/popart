#include <poponnx/error.hpp>
#include <poponnx/half.hpp>
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

TensorInfo StepIO::getTensorInfo(IArray &array) const {
  auto dtype = array.dataType();
  auto tRank = array.rank();
  std::vector<int64_t> shape;
  for (int i = 0; i < tRank; ++i) {
    shape.push_back(array.dim(i));
  }
  return TensorInfo(dtype, shape);
}

template <typename T>
T StepIO::get(TensorId id,
              const std::map<TensorId, IArray &> &M,
              std::string mapName) const {
  auto found = M.find(id);
  if (found == M.end()) {
    throw error("No tensor {} provided in CppStepIO's {}", id, mapName);
  }
  IArray &npArr = found->second;
  T stepData;
  stepData.data = npArr.data();
  stepData.info = getTensorInfo(npArr);
  return stepData;
}

ConstVoidData StepIO::in(TensorId id) const {
  return get<ConstVoidData>(id, inputs, "inputs");
}

MutableVoidData StepIO::out(TensorId id) const {
  return get<MutableVoidData>(id, outputs, "outputs");
}

} // namespace poponnx
