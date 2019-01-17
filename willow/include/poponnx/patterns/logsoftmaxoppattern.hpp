#ifndef GUARD_NEURALNET_LOGSOFTMAX_OP_PATTERN_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_OP_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>
#include <poponnx/patterns/sequenceexpander.hpp>

namespace poponnx {

// Replace LogSoftmax with
// logsoftmax(x) = log(softmax(x))
class LogSoftmaxOpPattern : public SequenceExpander {
public:
  bool matches(Op *) const override;
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }

private:
  // Replace the given op with the returned sequence of ops
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};

} // namespace poponnx

#endif
