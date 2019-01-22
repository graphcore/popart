#ifndef GUARD_NEURALNET_LOGSOFTMAX_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class NllLoss;

class LogSoftmaxOp : public ElementWiseUnaryOp {
public:
  LogSoftmaxOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::unique_ptr<Op> clone() const final;
};

// Has no grad op. LogSoftmaxOp pattern converts
// op into a sequence of log and softmax ops

} // namespace poponnx

#endif
