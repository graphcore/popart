#ifndef GUARD_NEURALNET_DIV_HPP
#define GUARD_NEURALNET_DIV_HPP

#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op/elementwise.hpp>

namespace poponnx {

// arg_0 / arg_1
class DivOp : public ElementWiseBinaryOp {
public:
  DivOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

// Base class for DivArg grad ops
class DivArgGradOp : public Op {
public:
  DivArgGradOp(const OperatorIdentifier &_opid,
               const std::vector<int64_t> &reduction_axes,
               const TensorInfo &forward_op_arg_info,
               const Op::Settings &settings_);
  void setup() final;
  static OutIndex getOutIndex() { return 0; }
  // In C = div(A,B) with numpy-style broadcasting,
  //   dA = reduceSum(div(dC,B)), and
  //   dB = reduceSum( negate( div( mul(dC, A), square(B) )) ).
  // this function returns the axes along which to perform the reduction.
  const std::vector<int64_t> &getReductionAxes() const;

private:
  // Used to set the outputs TensorInfo
  TensorInfo forward_op_arg_info;
  // reduction axes eventually passed to ReduceSumOp
  std::vector<int64_t> reduction_axes;
};

// gradOut / arg_1
class DivArg0GradOp : public DivArgGradOp {
public:
  DivArg0GradOp(const DivOp &, const std::vector<int64_t> &reduction_axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArg0InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

// - (gradOut * arg_0) / arg_1^2
class DivArg1GradOp : public DivArgGradOp {
public:
  DivArg1GradOp(const DivOp &, const std::vector<int64_t> &reduction_axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArg0InIndex() { return 1; }
  static InIndex getFwdArg1InIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
