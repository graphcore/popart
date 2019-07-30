#ifndef GUARD_NEURALNET_SPLIT_GRAD_OP_TO_CONCAT_PATTERN_HPP
#define GUARD_NEURALNET_SPLIT_GRAD_OP_TO_CONCAT_PATTERN_HPP

#include <popart/patterns/pattern.hpp>
#include <popart/patterns/sequenceexpander.hpp>

namespace popart {

// Replace ops that return their only input unchanged with an identity op
class SplitGradOpToConcatPattern : public SequenceExpander {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(Op *) const override;

private:
  // Replace the given op with the returned sequence of ops
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};
} // namespace popart

#endif
