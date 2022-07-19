// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SLICESTRUCT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SLICESTRUCT_HPP_

#include <cstdint>

namespace popart {

struct Slice {
  int64_t start;
  int64_t end;
  int64_t axis;
  bool flip;

  Slice(int64_t start_, int64_t end_, int64_t axis_, bool flip_)
      : start(start_), end(end_), axis(axis_), flip(flip_) {}

  Slice(int64_t start_, int64_t end_, int64_t axis_)
      : start(start_), end(end_), axis(axis_), flip(false) {}
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_SLICESTRUCT_HPP_
