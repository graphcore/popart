// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SLICESTRUCT_HPP
#define GUARD_NEURALNET_SLICESTRUCT_HPP

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

#endif
