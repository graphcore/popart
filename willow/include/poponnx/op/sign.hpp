#ifndef GUARD_NEURALNET_SIGN_HPP
#define GUARD_NEURALNET_SIGN_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class SignOp : public ElementWiseUnaryOp {
public:
  SignOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::unique_ptr<Op> clone() const final;

  static OperatorIdentifier getOpId(const Ir &ir);

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

// We use the tensorflow convention of defining the gradient to be 0 everywhere
// (including at 0).
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_grad.py#L801
class SignGradOp : public Op {
public:
  SignGradOp(const SignOp &);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  // TODO : T7052. SignGradOp does not need any inputs
  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace poponnx

#endif
