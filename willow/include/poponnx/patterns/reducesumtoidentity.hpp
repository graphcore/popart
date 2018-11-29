#ifndef GUARD_NEURALNET_REDUCE_SUM_TO_IDENTITY_PATTERN_HPP
#define GUARD_NEURALNET_REDUCE_SUM_TO_IDENTITY_PATTERN_HPP

#include <poponnx/patterns/pattern.hpp>

namespace poponnx {

// Replace reduce sum operations that reduce over no axes, or reduce on an axis
// of size 1, with an identity operation
class ReduceSumToIdentity : public Pattern {
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
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }
};
} // namespace poponnx

#endif
