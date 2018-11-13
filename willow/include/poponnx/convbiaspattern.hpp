#ifndef GUARD_NEURALNET_CONV_BIAS_PATTERN_HPP
#define GUARD_NEURALNET_CONV_BIAS_PATTERN_HPP

#include <poponnx/patterns.hpp>

namespace willow {

// Expand convolution operations that take a bias variable into a convolution
// operation followed by an add bias operation.
// conv(a, w, b) becomes addbias(conv(a, w), b)
class ConvBiasPattern : public Pattern {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(const Op *) const override;
  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  std::vector<const Tensor *> touches(const Op *) const override;
  // apply the pattern,
  // changes the graph of the op
  void apply(Op *) const override;
};
} // namespace willow

#endif
