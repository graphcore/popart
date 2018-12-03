#ifndef GUARD_NEURALNET_DIV_ARG_1_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_DIV_ARG_1_GRAD_OP_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace a DivArg1GradOp with
// (fwd_in1) -> [Square] -> (tmp1)
// {(gradOut), (fwd_in0)} -> [Mul] -> (tmp2)
// {(tmp2), (tmp1)} -> [Div] -> [Negate] -> [ReduceSum]
class DivArg1GradOpPattern : public Pattern {
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
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }
};

} // namespace poponnx

#endif
