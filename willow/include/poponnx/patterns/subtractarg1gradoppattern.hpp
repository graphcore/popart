#ifndef GUARD_NEURALNET_SUBTRACT_ARG1_GRAD_PATTERN_HPP
#define GUARD_NEURALNET_SUBTRACT_ARG1_GRAD_PATTERN_HPP

#include <poponnx/patterns/pattern.hpp>
#include <poponnx/patterns/sequenceexpander.hpp>

namespace poponnx {

// Replace a SubtractArg1GradOp with a negate followed by a reducesum
class SubtractArg1GradOpPattern : public SequenceExpander {
public:
  // Does op at the root of the
  // pattern make a match?
  bool matches(Op *) const override;
  // what phase should this Pattern run in? PRETOPOCONS, as it does not
  // handle topological constraints.
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }

private:
  // Replace the given op with the returned sequence of ops
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};

} // namespace poponnx

#endif
