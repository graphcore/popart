// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_INIT_ACCUMULATE_PATTERN_HPP
#define GUARD_NEURALNET_INIT_ACCUMULATE_PATTERN_HPP

#include <vector>
#include <popart/patterns/pattern.hpp>

namespace popart {
class Op;
class Tensor;

// Looks for this pattern where just one of the arguments to an
// ElementWiseBinaryOp is produced by an InitOp and where the
// output is consumed by only ElementWiseBinaryOps.
//
//  !InitOp-+
//          |-ElementWiseBinaryOp->ElementWiseBinaryOp
//   InitOp-+
//           ^^^^^^^^^^^^^^^^^^^
//
// Calls op->settings.inferTensorMappingToFrom.insert({initOpArg,otherOpArg})
// so that the argument from the InitOp will be laid out to match the other
// argument.
class InitAccumulatePattern : public PreAliasPattern {
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
