#ifndef GUARD_NEURALNET_ELEMENTWISEUNARY_HPP
#define GUARD_NEURALNET_ELEMENTWISEUNARY_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// Base class for elementwise unary operations
class ElementWiseUnaryOp : public Op {
public:
  ElementWiseUnaryOp(const OperatorIdentifier &_opid,
                     Ir *_ir,
                     const std::string &name = "",
                     const Attributes &_attr = {});
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

// Base class for gradients of elementwise, non linear, unary operations
// Non-linear elementwise ops gradients take both the input, and gradient
// output of the corresponding forward operation as inputs.
class ElementWiseNonLinearUnaryGradOp : public Op {
public:
  ElementWiseNonLinearUnaryGradOp(const OperatorIdentifier &_opid, Ir *_ir);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  // This grad op takes 2 inputs,
  // 1) the gradient of the output of the corresponding forward op, and
  // 2) the input of the forward op.
  // The indices at which these two tensors are inputs to this grad op are
  // getGradInIndex and getFwdArgInIndex respectively.
  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArgInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
