// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_POSTNREPL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_POSTNREPL_HPP_

#include <vector>
#include <popart/patterns/pattern.hpp>

namespace popart {
class Op;
class Tensor;

// consider,
// (ori) -> [*] -> {(rep1), (rep2), (rep3)}
// where rep1 = ori, rep2 = ori, rep3 = ori
// We call the Op [*] an N-replicator.
// It is similar to the identity in PreUniRepl, but it
// has N=3 copies of the input instead of N=1.
// if (ori) -> {[op0], [*], [op1]}, and
// (rep1) -> {[op0], [op2], [op2]}, and
// (rep2) -> {[op2], [op3]}
// (rep3) -> {}
// then this should be replaced by
// (ori) -> {[op0], [op0], [o1], [op2], [op2], [op2], [o3]}
// removals : [*], (rep1), (rep2), (rep3)
// [*] is the root of the pattern
// there are checks that the consumer dependecices of
// ori, rep1, rep2 and pre3 can be merged
class PostNRepl : public PreAliasPattern {
public:
  PostNRepl()           = default;
  ~PostNRepl() override = default;

  // AddGrad (where N = 2)
  // Pad with pad size zero (where N = 1) *
  // Sum with one input (where N = 1) *
  bool matches(Op *) const final;
  // rep1, rep2 and rep3 are touched (as they are deleted)
  // ori might be touched, if one its new consumers performs
  // an inplace modification to it.
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;

  // * : this pattern matches and removes []->()
  // whereas PreUniRepl matches and removes ()->[].
  // This pattern can be considered PostUniRepl when N = 1
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_POSTNREPL_HPP_
