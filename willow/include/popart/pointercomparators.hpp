// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POINTERCOMPARATORS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POINTERCOMPARATORS_HPP_

#include <memory>
#include <utility>
#include <vector>

namespace popart {
class Op;
class Tensor;

namespace popx {
class ICreatorCandidate;
} // namespace popx

using ICreatorCandidatePtr = std::shared_ptr<popx::ICreatorCandidate>;

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

struct VectorPTensorCmp {
  bool operator()(const std::vector<Tensor *> &a,
                  const std::vector<Tensor *> &b) const;
};

struct POpBoolCmp {
  bool operator()(const std::pair<Op *, bool> &a,
                  const std::pair<Op *, bool> &b) const;
};

struct POpIntCmp {
  bool operator()(std::pair<Op *, int> const &a,
                  std::pair<Op *, int> const &b) const;
};

struct PICreatorCandidateCmp {
  bool operator()(const popx::ICreatorCandidate *a,
                  const popx::ICreatorCandidate *b) const;
};

struct ICreatorCandidatePtrCmp {
  bool operator()(const ICreatorCandidatePtr a,
                  const ICreatorCandidatePtr b) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POINTERCOMPARATORS_HPP_
