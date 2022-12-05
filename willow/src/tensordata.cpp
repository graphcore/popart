// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
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

class TensorData::IData {
public:
  virtual void *data()                                                = 0;
  virtual std::size_t size()                                          = 0;
  virtual void doReset(const void *newSrc, const std::size_t newSize) = 0;
  virtual ~IData()                                                    = 0;
};
// Must always define pure virtual dtor as it will still get called
TensorData::IData::~IData() {}

class TensorData::OwningData final : public TensorData::IData {
public:
  OwningData(const void *src, const std::size_t size) {
    data_.resize(size);
    std::memcpy(data_.data(), src, size);
  }
  OwningData(std::vector<char> data) : data_(std::move(data)) {}

  void *data() override { return data_.data(); }
  std::size_t size() override { return data_.size(); }

  void doReset(const void *newSrc, const std::size_t newSize) override {
    data_.resize(newSize);
    std::memcpy(data_.data(), newSrc, newSize);
  }

  ~OwningData() override = default;

private:
  std::vector<char> data_;
};

class TensorData::NonOwningData final : public TensorData::IData {
public:
  NonOwningData(void *src, const std::size_t size_)
      : data_(src), size_(size_) {}

  void *data() override { return data_; }
  std::size_t size() override { return size_; }

  void doReset(const void *newSrc, const std::size_t newSize) override {
    if (size() != newSize) {
      // If newSize > size, you are doing a buffer overflow.
      // If newSize < size, there is dangling data on the end of the buffer.
      //
      // Maybe that is safe, but for now there is no use-case for this in
      // Popart so we just error.
      //
      // As of the time of writing, the only time that `size() != newSize`
      // occurs is in EnsureLossScaleFp32 transform, which changes an fp16 loss
      // scale tensor to be fp32, then overwrites the fp16 TensorData with an
      // fp32 value, hence a different `newSize`. The loss scale tensor has
      // OwningData though, so this is supportable. Generally, it is probably
      // better to avoid ever doing this in any other case, so make no attempt
      // to support it here.
      //
      // Note, we cannot make this safe by simply doing `data_ = newSrc`,
      // because we cannot change the pointer, because the Poplar streams are
      // directly connected to `data_`.
      throw internal_error(
          "Cannot overwrite the data of a TensorData::NonOwningData using data "
          "of different size (current = {}, new = {}).",
          size(),
          newSize);
    }
    std::memcpy(data_, newSrc, size());
  }

  ~NonOwningData() override = default;

private:
  void *data_;
  std::size_t size_;
};

TensorData::~TensorData()                           = default;
TensorData::TensorData(const TensorData &other)     = default;
TensorData::TensorData(TensorData &&other) noexcept = default;
TensorData &TensorData::operator=(const TensorData &other) = default;
TensorData &TensorData::operator=(TensorData &&other) noexcept = default;

TensorData::TensorData(std::shared_ptr<IData> data_)
    : data_(std::move(data_)) {}

TensorData TensorData::fromCopyOf(const void *src, const std::size_t size) {
  return TensorData(std::make_shared<OwningData>(src, size));
}
TensorData TensorData::fromViewOf(void *src, const std::size_t size) {
  return TensorData(std::make_shared<NonOwningData>(src, size));
}
TensorData TensorData::fromEmplaceOf(std::vector<char> &&data) {
  // In future could be generalised to other types too.
  // To help user, only binds to rvalue refs, as otherwise using this factory
  // instead of fromCopyOf is pointless.
  return TensorData(std::make_shared<OwningData>(std::move(data)));
}

void *TensorData::data() { return data_->data(); }
const void *TensorData::data() const { return data_->data(); }
std::size_t TensorData::size() const { return data_->size(); }

void TensorData::resetData(const ONNX_NAMESPACE::TensorProto &tp) {
  ConstVoidData cv_data = onnxutil::getConstData(tp);
  if (size() != cv_data.info.nbytes()) {
    throw error("cannot reset tensor data with data of non-matching size");
  }
  data_->doReset(cv_data.data, cv_data.info.nbytes());
}

void TensorData::resetData(const TensorInfo &info, const void *from) {
  if (size() != info.nbytes()) {
    throw error(
        "cannot reset tensor data with data of non-matching size {} vs {}",
        size(),
        info.nbytes());
  }
  data_->doReset(from, info.nbytes());
}

void TensorData::resetDataWithReplicaGrouping(const TensorInfo &info,
                                              const void *from,
                                              int numGroups) {
  auto deviceSize = (info.nbytes() * numGroups);
  if (size() != deviceSize) {
    throw error("cannot reset tensor data with data of non-matching size {} vs "
                "{} nbytes: {} number of groups: {}",
                size(),
                deviceSize,
                info.nbytes(),
                numGroups);
  }
  data_->doReset(from, deviceSize);
}

void TensorData::resetDataWithNonMatchingSize(const TensorInfo &info,
                                              const std::vector<char> from) {
  if (from.size() != info.nbytes()) {
    throw error("Size of supplied tensor data does not match expected size "
                "from TensorInfo");
  }
  data_->doReset(from.data(), info.nbytes());
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
