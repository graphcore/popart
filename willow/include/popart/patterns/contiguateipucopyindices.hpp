#ifndef GUARD_NEURALNET_CONTIGUATE_IP_COPY_INDICES_PATTERN_HPP
#define GUARD_NEURALNET_CONTIGUATE_IP_COPY_INDICES_PATTERN_HPP

#include <popart/patterns/pattern.hpp>
#include <popart/patterns/sequenceexpander.hpp>

namespace popart {

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

#endif
