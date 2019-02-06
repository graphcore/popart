#ifndef GUARD_NEURALNET_SQRT_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_SQRT_GRAD_OP_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

//      grad_in
//     ---------
//     2 sqrt(x)
//
//    sqrt(x) has already been done by the fwd pass
//    so this becomes...
//
//      grad_in
//     ---------
//     2 fwd_out
class SqrtGradOpPattern : public PreAliasPattern {
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
