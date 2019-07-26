#ifndef GUARD_NEURALNET_LOGSOFTMAX_OP_PATTERN_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>
#include <popart/patterns/sequenceexpander.hpp>

namespace popart {

// Replace LogSoftmax with
// logsoftmax(x) = log(softmax(x))
class LogSoftmaxOpPattern : public SequenceExpander {
public:
  bool matches(Op *) const override;

private:
  // Replace the given op with the returned sequence of ops
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};

} // namespace popart

#endif
