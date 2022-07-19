// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_NDINDICES_HPP_
#define POPART_WILLOW_INCLUDE_POPART_NDINDICES_HPP_

#include <cstdint>
#include <vector>

namespace popart {

class TensorInfo;

// A class for managing indexing into N-dimensional arrays/tensors
class NDIndices {

public:
  NDIndices(const TensorInfo &i);

  // convert a 1-dimensional index into a N-dimensional index,
  // assuming row-major indexing (right-most index is fastest)
  std::vector<int64_t> unflatten(int64_t rem) const;

  // the reverse of unflatten.
  // Special cases to aid with broadcasting:
  //   1) indices.size() > shape.size(), ignore left-most of indices
  //   2) indices.size() < shape.size(), bail
  //   3) indices[i] > shape[i], use indices % shape[i]
  //
  // Consider c = a + b with numpy-style broadcasting, the conditions
  // above allow us to pass an index of c into a and b and get the
  // correct index
  int64_t flatten(const std::vector<int64_t> &indices) const;

private:
  int64_t flatten_impl(const std::vector<int64_t> &indices) const;

  const TensorInfo &info;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_NDINDICES_HPP_
