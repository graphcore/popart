// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_RECIPROCALGRADOPPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_RECIPROCALGRADOPPATTERN_HPP_

#include "popart/patterns/pattern.hpp"
#include <vector>

namespace popart {
class Op;
class Tensor;

// Replace a ReciprocalGradOp with [Square] -> [Reciprocal] -> [Negate]
class ReciprocalGradOpPattern : public PreAliasPattern {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(Op *) const override;
  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  std::vector<const Tensor *> touches(Op *) const override;
  // apply the pattern,
  // changes the graph of the op
  bool apply(Op *) const override;
  // what phase should this Pattern run in? PRETOPOCONS, as it does not
  // handle topological constraints.
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_RECIPROCALGRADOPPATTERN_HPP_
