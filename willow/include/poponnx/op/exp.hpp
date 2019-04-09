#ifndef GUARD_NEURALNET_EXP_HPP
#define GUARD_NEURALNET_EXP_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class ExpOp : public ElementWiseUnaryOp {
public:
  ExpOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class ExpInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  ExpInplaceOp(const ExpOp &);
};

// Note that ExpGradOp does NOT
// follow the pattern of ElementWiseNonLinearUnaryGradOp
// because it takes the output of Exp as an input, and does
// not take the input of Exp as an input.
class ExpGradOp : public Op {
public:
  ExpGradOp(const ExpOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getGradInIndex() { return 0; }

  // The input index to this Op of the output of the Exp
  static InIndex getFwdOutInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return 0.1f; }
};

} // namespace poponnx

#endif
