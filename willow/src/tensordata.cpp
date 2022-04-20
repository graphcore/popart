// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <onnxutil.hpp>
#include <string>
#include <utility>
#include <vector>
#include <popart/error.hpp>
#include <popart/stepio.hpp>
#include <popart/tensordata.hpp>

#include "popart/logging.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/voiddata.hpp"

namespace onnx {
class TensorProto;
} // namespace onnx

namespace popart {

TensorData::TensorData(const ONNX_NAMESPACE::TensorProto &tp) {
  ConstVoidData cv_data = onnxutil::getConstData(tp);
  data_.resize(cv_data.info.nbytes());
  std::memcpy(data_.data(), cv_data.data, cv_data.info.nbytes());
}

TensorData::TensorData(const TensorInfo &info, const void *from)
    : TensorData(from, info.nbytes()) {}

TensorData::TensorData(const void *src, const size_t size) {
  data_.resize(size);
  std::memcpy(data_.data(), src, size);
}

void *TensorData::data() { return data_.data(); }
const void *TensorData::data() const { return data_.data(); }

void TensorData::resetData(const ONNX_NAMESPACE::TensorProto &tp) {
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

void TensorData::resetDataWithNonMatchingSize(const TensorInfo &info,
                                              const std::vector<char> from) {
  if (from.size() != info.nbytes()) {
    throw error("Size of supplied tensor data does not match expected size "
                "from TensorInfo");
  }
  data_.resize(info.nbytes());
  std::memcpy(data_.data(), from.data(), info.nbytes());
}

bool WeightsIO::contains(TensorId id) const {
  return weights.find(id) != weights.end();
}

MutableVoidData WeightsIO::weight(TensorId id) const {
  auto iter = weights.find(id);
  if (iter == weights.end()) {
    throw runtime_error("No TensorId {} in WeightsIO object", id);
  }
  return iter->second;
}

void WeightsIO::insert(TensorId id, MutableVoidData mvd) {

  auto iter = weights.find(id);
  if (iter != weights.end()) {
    throw runtime_error("TensorId {} already present in WeightsIO, cannot "
                        "insert");
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

ConstVoidData::ConstVoidData(const void *data_, const TensorInfo &info_)
    : data(data_), info(info_) {}

void ConstVoidData::store(std::vector<char> &&d, const TensorInfo &i) {
  optionalData    = d;
  data            = static_cast<const void *>(optionalData.data());
  hasOptionalData = true;
  info            = i;
}

} // namespace popart
