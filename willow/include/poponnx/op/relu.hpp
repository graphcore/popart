#ifndef GUARD_NEURALNET_RELU_HPP
#define GUARD_NEURALNET_RELU_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class ReluOp : public Op {
public:
  ReluOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;
};

class ReluInplaceOp : public Op {
public:
  ReluInplaceOp(const ReluOp &);
  void setup() final;
  // This in-place Op modifies its unique input at InIndex 0

  view::Region modifies(InIndex index) const final { return uses(index); }
  view::Region aliases(InIndex index) const final { return uses(index); }
  // "uses" is still the full region
  // "fwdRegMap" and "bwdRegMap" are still the identity
};

// takes output of ReluOp as input and not the input of ReluOp
// to determine where gradients become zero. It might be better
// (depending in what can be in-placed) to rather take the input
// of ReluOp in to do this (or a boolean tensor).
class ReluGradOp : public Op {
public:
  ReluGradOp(const ReluOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // The index at which the output of the Relu (the "relud" tensor)
  // is an input to this ReluGradOp
  static InIndex getReludInIndex() { return 1; }

  // The index at which the gradient of the output of
  // the Relu is an input to this ReluGradOp
  static InIndex getGradReludInIndex() { return 0; }

  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
