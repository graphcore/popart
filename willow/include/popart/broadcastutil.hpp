// Copyright (c) 2020, Graphcore Ltd, All rights reserved

#ifndef GUARD_NEURALNET_BROADCAST_UTILS_HPP
#define GUARD_NEURALNET_BROADCAST_UTILS_HPP

#include <vector>

namespace popart {

template <typename T>
static std::vector<T>
padShape(const std::vector<T> &shape, size_t padded_size, T pad_value) {
  std::vector<T> result(padded_size - shape.size(), pad_value);
  result.insert(result.end(), shape.begin(), shape.end());
  return result;
}

template <typename T>
static std::vector<T> unpadShape(const std::vector<T> &shape,
                                 size_t unpadded_size) {
  std::vector<T> result;
  auto offset = shape.size() - unpadded_size;
  result.insert(result.begin(), shape.begin() + offset, shape.end());
  return result;
}

} // namespace popart

#endif
