#ifndef GUARD_NEURALNET_VARIADIC_HPP
#define GUARD_NEURALNET_VARIADIC_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// A Variadic Op is a reduction which applies an element-wise binary op
// sequentially to a variable number of inputs, 'til 1 tensor remains
class VariadicOp : public Op {
public:
  VariadicOp(const OperatorIdentifier &_opid, const Op::Settings &settings);

  // One grad Op per input, and all grad Ops are of the same type
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  // The output shape is determined by, starting with the first input,
  // iteratively applying the numpy-broadcasting rules on the sequence
  void setup() final;

  // VariadicOps have a variable number of inputs, but exactly one output
  static OutIndex getOutIndex() { return 0; }

  bool canBeReplacedByIdentity() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  // return the gradient Op for the i'th input, 0 <= i < number of inputs
  virtual std::unique_ptr<Op> getIthGrad(int) const = 0;
};

class VariadicGradOp : public Op {
public:
  VariadicGradOp(const OperatorIdentifier &_opid, const VariadicOp &, InIndex);

  // This Op creates the gradient for input fwdIndex
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  static InIndex getGradInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
  InIndex getFwdIndex() { return fwdIndex; }
  const TensorInfo &getFwdInputInfo() { return fwdInputInfo; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  InIndex fwdIndex;
  // the info corresponding to tensor at index fwdIndex to the forward op
  TensorInfo fwdInputInfo;
  std::map<int, int> gradOutToNonGradInInfo;
};

class NonLinearVariadicGradOp : public VariadicGradOp {
public:
  NonLinearVariadicGradOp(const OperatorIdentifier &_opid,
                          const VariadicOp &,
                          InIndex);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  static InIndex getFwdInIndex() { return 1; }
  static InIndex getFwdOutInIndex() { return 2; }

private:
  std::vector<GradInOutMapper> gradInputInfoVec;
};

class LinearVariadicGradOp : public VariadicGradOp {
public:
  LinearVariadicGradOp(const OperatorIdentifier &_opid,
                       const VariadicOp &,
                       InIndex);

  virtual bool hasScale() const { return false; }
  virtual float getScale() const { return 1.0; }
};

} // namespace poponnx

#endif
