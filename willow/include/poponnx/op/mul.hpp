#ifndef GUARD_NEURALNET_MUL_HPP
#define GUARD_NEURALNET_MUL_HPP

#include <vector>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/reducesum.hpp>

namespace poponnx {

class MulOp : public Op {
public:
  MulOp(const OperatorIdentifier &_opid,
        Ir *_ir,
        const std::string &name = "",
        const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Current implementation places arg0 input at index 0, and arg1 input
  // at index 1.
  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

class MulArgGradOp : public Op {
public:
  MulArgGradOp(const OperatorIdentifier &_opid,
               Ir *_ir,
               const std::vector<int64_t> &reduction_axes,
               const TensorInfo &forward_op_arg_info);
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

private:
  std::vector<int64_t> reduction_axes;
  TensorInfo forward_op_arg_info;
};

class MulArg0GradOp : public MulArgGradOp {
public:
  MulArg0GradOp(MulOp *, const std::vector<int64_t> &reduction_axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  static OutIndex getOutIndex() { return 0; }
};

class MulArg1GradOp : public MulArgGradOp {
public:
  MulArg1GradOp(MulOp *, const std::vector<int64_t> &reduction_axes);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace poponnx

#endif
