#ifndef GUARD_NEURALNET_TAN_TO_SIN_OVER_COS_PATTERN_HPP
#define GUARD_NEURALNET_TAN_TO_SIN_OVER_COS_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace a TanOp with
// (fwd_in) -> [Cos] -> (cos_out)
// (fwd_in) -> [Sin] -> (sin_out)
// {(sin_out), (cos_out) -> [Div] -> (fwd_out)
class TanToSinOverCosPattern : public Pattern {
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
