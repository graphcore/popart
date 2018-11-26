#ifndef GUARD_NEURALNET_CONV_BIAS_PATTERN_HPP
#define GUARD_NEURALNET_CONV_BIAS_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace willow {

// Expand convolution operations that take a bias variable into a convolution
// operation followed by an add bias operation.
// conv(a, w, b) becomes addbias(conv(a, w), b)
class ConvBiasPattern : public Pattern {
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
  void apply(Op *) const override;
  // what phase should this Pattern run in? PRETOPOCONS, as it does not
  // handle topological constraints.
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }
};
} // namespace willow

#endif
