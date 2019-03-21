#ifndef GUARD_NEURALNET_CONV_DATA_GRAD_PATTERN_HPP
#define GUARD_NEURALNET_CONV_DATA_GRAD_PATTERN_HPP

#include <poponnx/patterns/pattern.hpp>

namespace poponnx {

// Expand convolution the conv data grad to a X/Y flip of the weights and then a
// convolution
class ConvDataGradPattern : public PreAliasPattern {
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
