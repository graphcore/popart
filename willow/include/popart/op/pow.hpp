#ifndef GUARD_NEURALNET_POW_HPP
#define GUARD_NEURALNET_POW_HPP

#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

// arg_0 / arg_1
class PowOp : public ElementWiseBinaryOp {
public:
  PowOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static OperatorIdentifier getOpId(const Ir &ir);
};

// Base class for PowArg grad ops
class PowArgGradOp : public Op {
public:
  PowArgGradOp(const OperatorIdentifier &_opid,
               const std::vector<int64_t> &reduction_axes,
               const TensorInfo &forward_op_arg_info,
               const Op::Settings &settings_);
  void setup() final;
  static OutIndex getOutIndex() { return 0; }
  // In C = pow(A,B) =  A ** B with numpy-style broadcasting,
  //   dA = reduce_sum(mul(B, pow(A, B-1)))
  //   dB = reduce_sum(mul(C, log(A)))
  // this function returns the axes along which to perform the reduction.
  const std::vector<int64_t> &getReductionAxes() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  // Used to set the outputs TensorInfo
  TensorInfo forward_op_arg_info;
  // reduction axes eventually passed to ReduceSumOp
  std::vector<int64_t> reduction_axes;
};

class PowArg0GradOp : public PowArgGradOp {
public:
  PowArg0GradOp(const PowOp &, const std::vector<int64_t> &reduction_axes);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArg0InIndex() { return 1; }
  static InIndex getFwdArg1InIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }
};

class PowArg1GradOp : public PowArgGradOp {
public:
  PowArg1GradOp(const PowOp &, const std::vector<int64_t> &reduction_axes);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArg0InIndex() { return 1; }
  static InIndex getFwdOutIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
