#ifndef GUARD_NEURALNET_MATMUL_HPP
#define GUARD_NEURALNET_MATMUL_HPP

#include <poponnx/ir.hpp>

namespace willow {

class MatMulOp : public Op {
public:
  MatMulOp(const onnx::NodeProto &node, Ir *pir);
  MatMulOp(const MatMulOp &) = default;
  MatMulOp &operator=(const MatMulOp &) = delete;
  virtual ~MatMulOp() override          = default;

  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
  virtual std::unique_ptr<Op> clone() const override final;

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
  virtual ~MatMulLhsGradOp() override                 = default;

  static int getGradInputIndex() { return 0; }
  static int getRhsInputIndex() { return 1; }

  virtual void setup() override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;

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
  virtual ~MatMulRhsGradOp() override                 = default;

  static int getGradInputIndex() { return 0; }
  static int getLhsInputIndex() { return 1; }

  virtual void setup() override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;

private:
  Shape outputShape;
  TensorInfo fwdOpOutputGrad;
  TensorInfo lhs;
};

} // namespace willow

#endif
