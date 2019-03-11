#ifndef GUARD_NEURALNET_ELEMENTWISEUNARY_HPP
#define GUARD_NEURALNET_ELEMENTWISEUNARY_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// Base class for elementwise unary operations
class ElementWiseUnaryOp : public Op {
public:
  ElementWiseUnaryOp(const OperatorIdentifier &_opid,
                     const Op::Settings &settings);
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

// Base class for gradients of elementwise, non linear, unary operations
// Non-linear elementwise ops gradients take both the input, and gradient
// output of the corresponding forward operation as inputs.
class ElementWiseNonLinearUnaryGradOp : public Op {
public:
  ElementWiseNonLinearUnaryGradOp(const OperatorIdentifier &_opid,
                                  const ElementWiseUnaryOp &fwdOp);
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

// Base class for elementwise binary operations
class ElementWiseBinaryOp : public Op {
public:
  ElementWiseBinaryOp(const OperatorIdentifier &_opid,
                      const Op::Settings &settings);
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
