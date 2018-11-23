#ifndef GUARD_NEURALNET_MATMUL_HPP
#define GUARD_NEURALNET_MATMUL_HPP

#include <poponnx/ir.hpp>

namespace willow {

class MatMulOp : public Op {
public:
  MatMulOp(const onnx::NodeProto &node, Ir *pir);
  MatMulOp(const MatMulOp &) = default;
  MatMulOp &operator=(const MatMulOp &) = delete;
  ~MatMulOp() override                  = default;

  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static int getLhsInputIndex() { return 0; }
  static int getRhsInputIndex() { return 1; }
  static int getOutputIndex() { return 0; }

  const Tensor *lhsIn() const;
  const Tensor *rhsIn() const;
  const Tensor *out() const;

private:
  Shape outputShape;
};

class MatMulLhsGradOp : public Op {
public:
  MatMulLhsGradOp(const MatMulOp &op_);
  MatMulLhsGradOp(const MatMulLhsGradOp &) = default;
  MatMulLhsGradOp &operator=(const MatMulLhsGradOp &) = delete;
  ~MatMulLhsGradOp() override                         = default;

  static int getGradInputIndex() { return 0; }
  static int getRhsInputIndex() { return 1; }

  void setup() final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  Shape outputShape;
  TensorInfo fwdOpOutputGrad;
  TensorInfo rhs;
};

class MatMulRhsGradOp : public Op {
public:
  MatMulRhsGradOp(const MatMulOp &op_);
  MatMulRhsGradOp(const MatMulRhsGradOp &) = default;
  MatMulRhsGradOp &operator=(const MatMulRhsGradOp &) = delete;
  ~MatMulRhsGradOp() override                         = default;

  static int getGradInputIndex() { return 0; }
  static int getLhsInputIndex() { return 1; }

  void setup() final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

private:
  Shape outputShape;
  TensorInfo fwdOpOutputGrad;
  TensorInfo lhs;
};

} // namespace willow

#endif
