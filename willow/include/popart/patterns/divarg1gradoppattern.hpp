// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DIV_ARG_1_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_DIV_ARG_1_GRAD_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace a DivArg1GradOp with
// (fwd_in1) -> [Square] -> (tmp1)
// {(grad_in), (fwd_in0)} -> [Mul] -> (tmp2)
// {(tmp2), (tmp1)} -> [Div] -> [Negate] -> [ReduceSum] -> (grad_out)
class DivArg1GradOpPattern : public PreAliasPattern {
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

#endif
