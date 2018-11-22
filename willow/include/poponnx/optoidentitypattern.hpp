#ifndef GUARD_NEURALNET_REDUCE_SUM_TO_IDENTITY_PATTERN_HPP
#define GUARD_NEURALNET_REDUCE_SUM_TO_IDENTITY_PATTERN_HPP

#include <poponnx/patterns.hpp>

namespace willow {

// Replace ops that return their only input unchanged with an identity op
class OpToIdentityPattern : public Pattern {
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
};
} // namespace willow

#endif
