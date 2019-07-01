#ifndef GUARD_NEURALNET_POW_ARG_1_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_POW_ARG_1_GRAD_OP_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace a PowArg1GradOp with
// (fwd_in0) -> [Log] -> (tmp1)
// {(out), (tmp1)} -> [Mult] -> [ReduceSum] -> (grad_out)
class PowArg1GradOpPattern : public PreAliasPattern {
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

} // namespace poponnx

#endif
