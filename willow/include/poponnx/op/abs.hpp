#ifndef GUARD_NEURALNET_ABS_HPP
#define GUARD_NEURALNET_ABS_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class AbsOp : public ElementWiseUnaryOp {
public:
  AbsOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class AbsGradOp : public Op {
public:
  AbsGradOp(const AbsOp &);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArgInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
  virtual float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace poponnx

#endif
