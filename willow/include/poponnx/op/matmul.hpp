#ifndef GUARD_NEURALNET_MATMUL_HPP
#define GUARD_NEURALNET_MATMUL_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class MatMulOp : public Op {
public:
  MatMulOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  MatMulOp(const MatMulOp &) = default;
  MatMulOp &operator=(const MatMulOp &) = delete;
  ~MatMulOp() override                  = default;

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getLhsInIndex() { return 0; }
  static InIndex getRhsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  const Tensor *lhsIn() const;
  const Tensor *rhsIn() const;
  const Tensor *out() const;

  // The ONNX tensor shape
  Shape lhsBroadcastShape() const;
  Shape rhsBroadcastShape() const;

  // Follow the numpy matmul broadcasting rules for the left operand shape
  static Shape lhsNpBroadcastShape(Shape lhs, Shape rhs);

  // Follow the numpy matmul broadcasting rules for the right operand shape
  static Shape rhsNpBroadcastShape(Shape lhs, Shape rhs);

  // Follow the numpy matmul broadcasting rules for the output shape
  static Shape npMatMulOut(Shape lhs, Shape rhs);
};

class MatMulLhsGradOp : public Op {
public:
  MatMulLhsGradOp(const MatMulOp &op_);
  MatMulLhsGradOp(const MatMulLhsGradOp &) = default;
  MatMulLhsGradOp &operator=(const MatMulLhsGradOp &) = delete;
  ~MatMulLhsGradOp() override                         = default;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getRhsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void setup() final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  // The ONNX tensor shape
  // The shape of the grad op's gradient input
  Shape getGradInputShape() const;
  // The shape of the grad op's rhs input
  Shape getRhsInputShape() const;
  // The shape of the grad op's output
  Shape getOutputShape() const;

private:
  TensorInfo fwdOpOutputGrad;
  TensorInfo fwdOpLhsInfo;
  TensorInfo fwdOpRhsInfo;
};

class MatMulRhsGradOp : public Op {
public:
  MatMulRhsGradOp(const MatMulOp &op_);
  MatMulRhsGradOp(const MatMulRhsGradOp &) = default;
  MatMulRhsGradOp &operator=(const MatMulRhsGradOp &) = delete;
  ~MatMulRhsGradOp() override                         = default;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getLhsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void setup() final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  // The ONNX tensor shape
  // The shape of the grad op's gradient input
  Shape getLhsInputShape() const;
  // The shape of the grad op's rhs input
  Shape getGradInputShape() const;
  // The shape of the grad op's output
  Shape getOutputShape() const;

private:
  TensorInfo fwdOpOutputGrad;
  TensorInfo fwdOpLhsInfo;
  TensorInfo fwdOpRhsInfo;
};

} // namespace poponnx

#endif
