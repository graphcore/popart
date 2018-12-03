#ifndef GUARD_NEURALNET_DIV_HPP
#define GUARD_NEURALNET_DIV_HPP

#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>

namespace poponnx {

// arg_0 / arg_1
class DivOp : public Op {
public:
  DivOp(const onnx::NodeProto &node, Ir *pir);
  DivOp(const OpConstructorBundle &);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

// Base class for DivArg grad ops
class DivArgGradOp : public Op {
public:
  DivArgGradOp(const OpConstructorBundle &,
               const std::vector<int64_t> &reduction_axes,
               const TensorInfo &forward_op_arg_info);
  void setup() final;
  static OutIndex getOutIndex() { return 0; }
  // In C = div(A,B) with numpy-style broadcasting,
  //   dA = reduceSum(div(dC,B)), and
  //   dB = reduceSum( negate( div( mul(dC, A), square(B) )) ).
  // this function returns the axes along which to perform the reduction.
  const std::vector<int64_t> &getReductionAxes() const;

private:
  TensorInfo forward_op_arg_info;
  std::vector<int64_t> reduction_axes;
};

// gradOut / arg_1
class DivArg0GradOp : public DivArgGradOp {
public:
  DivArg0GradOp(DivOp *, const std::vector<int64_t> &reduction_axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

// - (gradOut * arg_0) / arg_1^2
class DivArg1GradOp : public DivArgGradOp {
public:
  DivArg1GradOp(DivOp *, const std::vector<int64_t> &reduction_axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace poponnx

#endif
