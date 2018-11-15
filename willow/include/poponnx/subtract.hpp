#ifndef GUARD_NEURALNET_SUBTRACT_HPP
#define GUARD_NEURALNET_SUBTRACT_HPP

#include <poponnx/identity.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/negate.hpp>

namespace willow {

class SubtractOp : public Op {
public:
  SubtractOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static int arg0Index();
  static int arg1Index();
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class SubtractArg0GradOp : public IdentityOp {
public:
  SubtractArg0GradOp(SubtractOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

// TODO (task T5432) should inherit from ReduceSum when we have numpy
// broadcasting
class SubtractArg1GradOp : public NegateOp {
public:
  SubtractArg1GradOp(SubtractOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace willow

#endif
