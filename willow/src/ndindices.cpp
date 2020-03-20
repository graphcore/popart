// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ndindices.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

NDIndices::NDIndices(const TensorInfo &i) : info(i) {}

std::vector<int64_t> NDIndices::unflatten(int64_t rem) const {
  std::vector<int64_t> indices(info.shape());
  for (int i = info.rank() - 1; i >= 0; i--) {
    auto d     = info.dim(i);
    indices[i] = rem % d;
    rem /= d;
  }
  if (rem != 0) {
    throw error("index too large in unflatten");
  }
  return indices;
}

int64_t NDIndices::flatten(const std::vector<int64_t> &indices) const {
  // case 2) bailing
  if (indices.size() < info.rank()) {
    throw error("too few indices in flatten argument");
  }
  // case 1) ignoring left-most (slowest) indices of indices
  else if (indices.size() > info.rank()) {
    return flatten_impl(
        std::vector<int64_t>(indices.end() - info.rank(), indices.end()));
  } else {
    return flatten_impl(indices);
  }
}

int64_t NDIndices::flatten_impl(const std::vector<int64_t> &indices) const {
  int64_t stride = info.nelms();
  int64_t index  = 0l;
  for (int d = 0; d < info.rank(); ++d) {
    stride /= info.dim(d);
    // 3) modulo arithmetic
    index += (indices[d] % info.dim(d)) * stride;
  }
  if (index > info.nelms()) {
    throw internal_error("In flatten : final index too large");
  }
  return index;
}

} // namespace popart
