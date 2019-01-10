#ifndef GUARD_NEURALNET_LOGSOFTMAX_OP_PATTERN_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_OP_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace LogSoftmax with
// logsoftmax(x) = log(softmax(x))
class LogSoftmaxOpPattern : public Pattern {
public:
  bool matches(Op *) const override;
  // std::vector<const Tensor *> touches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override { return {}; }
  bool apply(Op *) const override;
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }
};

} // namespace poponnx

#endif
