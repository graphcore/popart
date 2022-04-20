// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VOIDDATA_HPP_HPP
#define GUARD_NEURALNET_VOIDDATA_HPP_HPP

#include <vector>
#include <popart/tensorinfo.hpp>

namespace popart {

/// A class to point to constant data.
class ConstVoidData {
public:
  ConstVoidData() = default;
  ConstVoidData(const void *data_, const TensorInfo &info_);

  const void *data = nullptr;
  // This is used to confirm that data is as expected
  TensorInfo info;

  bool storesData() const { return hasOptionalData; }
  void store(std::vector<char> &&d, const TensorInfo &i);

private:
  std::vector<char> optionalData;
  bool hasOptionalData{false};
};

/// A class to point to non-constant data.
class MutableVoidData {
public:
  void *data = nullptr;
  // This is used to confirm that data is as expected
  TensorInfo info;
};

} // namespace popart

#endif
