// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_CONTIGUATEIPUCOPYINDICES_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_CONTIGUATEIPUCOPYINDICES_HPP_

#include <vector>
#include <popart/patterns/pattern.hpp>

namespace popart {
class Op;
class Tensor;

// Note 1:
// This Pattern is similar to SequenceExpander, but different enough to merit
// its own standalone implementation.
//
// Note 2:
// This Pattern should not be run unless explicity required by the user, it is
// currently only used to enable pipelining with non-contiguous IPUCopys
//
class ContiguateIpuCopyIndicesPattern : public PreAliasPattern {
public:
  // All IpuCopyOps with a single source IPU, for which delta is not +-1
  bool matches(Op *) const final;

  // return {}
  std::vector<const Tensor *> touches(Op *) const final;

  // Replace the long-range, discontiguous copy,
  // firstIpuId -> finalIpuId
  //
  // with a seqeuce of contiguous copies,
  //
  // firstIpuId ->
  // firstIpuId + delta ->
  // firstIpuId + 2*delta ->
  // ... -> finalIpuId,
  //
  // where delta is +1 or -1.
  //
  bool apply(Op *) const final;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_CONTIGUATEIPUCOPYINDICES_HPP_
