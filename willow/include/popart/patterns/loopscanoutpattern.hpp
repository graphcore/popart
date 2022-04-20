// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOOP_SCAN_OUT_PATTERN_HPP
#define GUARD_NEURALNET_LOOP_SCAN_OUT_PATTERN_HPP

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

// Replaces implicit scan outputs in loops with explicit dynamic updates
// Implicit scan outputs are outputs added to the LoopOp which do not correspond
// to a loop carried dependency input.
// The LoopOp will stack implicit scan outputs on a new axis with the same size
// as the maximum number of loop iterations (n), e.g. out = [iteration 0 output,
// iteration 1 output, ..., iteration n output]
class LoopScanOutPattern : public PreAliasPattern {
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
};

} // namespace popart

#endif
