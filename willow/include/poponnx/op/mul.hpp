#ifndef GUARD_NEURALNET_MUL_HPP
#define GUARD_NEURALNET_MUL_HPP

#include <vector>
#include <poponnx/names.hpp>
#include <poponnx/op/elementwise.hpp>
#include <poponnx/op/reducesum.hpp>

namespace poponnx {

class MulOp : public ElementWiseBinaryOp {
public:
  MulOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static OperatorIdentifier getOpId(const Ir &ir);
};

class MulArgGradOp : public Op {
public:
  MulArgGradOp(const OperatorIdentifier &_opid,
               const std::vector<int64_t> &reduction_axes,
               const TensorInfo &forward_op_arg_info,
               const Op::Settings &settings_);
  // MulArgGradOp(const OpConstructorBundle &,
  // const std::vector<int64_t> &reduction_axes,
  // const TensorInfo &forward_op_arg_info);
  void setup() final;
  // In C = mul(A,B) with numpy-style broadcasting,
  //   dA = reduceSum(mul(dC,B)), and
  //   dB = reduceSum(mul(dC,A)).
  // this function returns the axes along which to perform the reduction.
  const std::vector<int64_t> &getReductionAxes();
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  std::vector<int64_t> reduction_axes;
  TensorInfo forward_op_arg_info;
};

class MulArg0GradOp : public MulArgGradOp {
public:
  MulArg0GradOp(const MulOp &, const std::vector<int64_t> &reduction_axes);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  static OutIndex getOutIndex() { return 0; }
};

class MulArg1GradOp : public MulArgGradOp {
public:
  MulArg1GradOp(const MulOp &, const std::vector<int64_t> &reduction_axes);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace poponnx

#endif
