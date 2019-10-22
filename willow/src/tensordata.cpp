#include <popart/error.hpp>
#include <popart/half.hpp>
#include <popart/onnxutil.hpp>
#include <popart/tensordata.hpp>

#include <cstring>

namespace popart {

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
const void *TensorData::data() const { return data_.data(); }

void TensorData::resetData(const onnx::TensorProto &tp) {
  ConstVoidData cv_data = onnxutil::getConstData(tp);
  if (data_.size() != cv_data.info.nbytes()) {
    throw error("cannot reset tensor data with data of non-matching size");
  }
  std::memcpy(data_.data(), cv_data.data, cv_data.info.nbytes());
}

void TensorData::resetData(const TensorInfo &info, const void *from) {
  if (data_.size() != info.nbytes()) {
    throw error("cannot reset tensor data with data of non-matching size");
  }
  data_.resize(info.nbytes());
  std::memcpy(data_.data(), from, info.nbytes());
}

StepIO::StepIO(std::map<TensorId, IArray &> inputs,
               std::map<TensorId, IArray &> outputs) {
  for (auto p : inputs) {
    inputsInfo.insert({p.first, {p.second, 0}});
  }

  for (auto p : outputs) {
    outputsInfo.insert({p.first, {p.second, 0}});
  }
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
              std::map<TensorId, ArrayInfo> &M,
              int64_t numElements,
              bool advanceIn,
              std::string mapName) {

  auto found = M.find(id);
  if (found == M.end()) {
    throw error("No tensor {} provided in PyStepIO's {}", id, mapName);
  }

  ArrayInfo &arrayInfo = found->second;
  auto offset          = arrayInfo.offset;

  T stepData;
  stepData.info = getTensorInfo(arrayInfo.array);

  // Set the data using the offset

  stepData.data = static_cast<uint8_t *>(arrayInfo.array.data()) + offset;

  if (advanceIn) {

    auto numBytes = stepData.info.getDataTypeInfo()->nbytes() * numElements;

    // Wrap around if we read all the data
    if (offset + numBytes == stepData.info.nbytes()) {
      arrayInfo.offset = 0;
    } else {
      arrayInfo.offset = offset + numBytes;
    }
  }

  return stepData;
}

template <typename T>
void StepIO::advance(TensorId id,
                     std::map<TensorId, ArrayInfo> &M,
                     int64_t numElements,
                     std::string mapName) {

  auto found = M.find(id);
  if (found == M.end()) {
    throw error("No tensor {} provided in PyStepIO's {}", id, mapName);
  }

  ArrayInfo &arrayInfo = found->second;
  auto offset          = arrayInfo.offset;

  T stepData;
  stepData.info = getTensorInfo(arrayInfo.array);

  auto numBytes = stepData.info.getDataTypeInfo()->nbytes() * numElements;

  if (offset + numBytes == stepData.info.nbytes()) {
    arrayInfo.offset = 0;
  } else {
    arrayInfo.offset = offset + numBytes;
  }
}

ConstVoidData StepIO::in(TensorId id, int64_t numElements, bool /*prefetch*/) {
  return get<ConstVoidData>(id, inputsInfo, numElements, false, "inputs");
}

void StepIO::inComplete(TensorId id, int64_t numElements) {
  return advance<ConstVoidData>(id, inputsInfo, numElements, "inputs");
}

MutableVoidData StepIO::out(TensorId id, int64_t numElements) {
  return get<MutableVoidData>(id, outputsInfo, numElements, true, "outputs");
}

bool WeightsIO::contains(TensorId id) const {
  return weights.find(id) != weights.end();
}

MutableVoidData WeightsIO::weight(TensorId id) const {
  auto iter = weights.find(id);
  if (iter == weights.end()) {
    throw error("No TensorId {} in WeightsIO object", id);
  }
  return iter->second;
}

void WeightsIO::insert(TensorId id, MutableVoidData mvd) {

  auto iter = weights.find(id);
  if (iter != weights.end()) {
    throw error("TensorId {} already present in WeightsIO, cannot insert");
  }
  weights.insert({id, mvd});
}

ConstVoidData StepIOCallback::in(TensorId id, int64_t, bool prefetch) {
  return inputCb(id, prefetch);
}

void StepIOCallback::inComplete(TensorId id, int64_t) {
  return inputCompleteCb(id);
}

MutableVoidData StepIOCallback::out(TensorId id, int64_t) {
  return outputCb(id);
}
void StepIOCallback::outComplete(TensorId id) { return outputCompleteCb(id); }
} // namespace popart
