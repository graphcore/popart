#ifndef GUARD_NEURALNET_RELU_HPP
#define GUARD_NEURALNET_RELU_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class ReluOp : public Op {
public:
  ReluOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  bool hasInplaceVariant(InIndex) const final;
  std::unique_ptr<Op> getInplaceVariant(InIndex) final;
};

class ReluInplaceOp : public Op {
public:
  ReluInplaceOp(ReluOp *);
  void setup() final;
  // This in-place Op modifies its unique input at InIndex 0
  bool modifies(InIndex) const final;
};

// takes output of ReluOp as input and not the input of ReluOp
// to determine where gradients become zero. It might be better
// (depending in what can be in-placed) to rather take the input
// of ReluOp in to do this (or a boolean tensor).
class ReluGradOp : public Op {
public:
  ReluGradOp(ReluOp *);
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
