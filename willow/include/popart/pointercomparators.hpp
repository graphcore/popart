// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POINTERCOMPARATORS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POINTERCOMPARATORS_HPP_

#include <utility>

namespace popart {
class Op;
class Tensor;

// A note on non-determinism. For maps with
// pointers as keys, iterating through them
// is non-deterministic with the default comparator.
/// To prevent non-determinism, POpCmp is used on any sets and maps that use
/// pointers to operators as a set/map key.
struct POpCmp {
  bool operator()(const Op *a, const Op *b) const;
};

struct PTensorCmp {
  bool operator()(const Tensor *const &a, const Tensor *const &b) const;
};

struct POpBoolCmp {
  bool operator()(const std::pair<Op *, bool> &a,
                  const std::pair<Op *, bool> &b) const;
};

struct POpIntCmp {
  bool operator()(std::pair<Op *, int> const &a,
                  std::pair<Op *, int> const &b) const;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POINTERCOMPARATORS_HPP_
