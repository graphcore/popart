#ifndef GUARD_NEURALNET_ELEMENTWISEUNARY_HPP
#define GUARD_NEURALNET_ELEMENTWISEUNARY_HPP

#include <popart/op.hpp>

namespace popart {

// Base class for elementwise unary operations
class ElementWiseUnaryOp : public Op {
public:
  ElementWiseUnaryOp(const OperatorIdentifier &_opid,
                     const Op::Settings &_settings);
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  // Making this function override and not final, as there
  // may be a more / less expensive to compute non-linearity.
  float getSubgraphValue() const override { return getLowSubgraphValue(); }
};

class ElementWiseInplaceUnaryOp : public ElementWiseUnaryOp {
public:
  ElementWiseInplaceUnaryOp(const OperatorIdentifier &_opid,
                            const Op::Settings &_settings)
      : ElementWiseUnaryOp(_opid, _settings) {}

  view::Region modifies(InIndex index) const final { return uses(index); }
  view::Region aliases(InIndex index) const final { return uses(index); }
  // "uses" is still the full region
  // "fwdRegMap" is still the identity
  // "bwdRegMap" is still the identity
};

// Base class for gradients of element-wise, non-linear, unary operations
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

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

// Base class for elementwise binary operations
class ElementWiseBinaryOp : public Op {
public:
  ElementWiseBinaryOp(const OperatorIdentifier &_opid,
                      const Op::Settings &_settings);
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

// Base class for comparison operations
class BinaryComparisonOp : public Op {
public:
  BinaryComparisonOp(const OperatorIdentifier &_opid,
                     const Op::Settings &_settings);
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
