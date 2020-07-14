// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SLICESTRUCT_HPP
#define GUARD_NEURALNET_SLICESTRUCT_HPP

#include <cstdint>

namespace popart {

struct Slice {
  int64_t start;
  int64_t end;
  int64_t axis;

  Slice(int64_t start_, int64_t end_, int64_t axis_)
      : start(start_), end(end_), axis(axis_) {}
};

} // namespace popart

#endif
