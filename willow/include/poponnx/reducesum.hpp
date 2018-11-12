#ifndef GUARD_NEURALNET_REDUCESUM_HPP
#define GUARD_NEURALNET_REDUCESUM_HPP

#include <poponnx/ir.hpp>

namespace willow {

class ReduceSumOp : public Op {
public:
  ReduceSumOp(const OpConstructorBundle &);
  ReduceSumOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override final;
  void setup() override final;

  // A list of integers, along which to reduce. These axes will either be
  // removed or have size 1, depending on the value of getKeepDims.
  const std::vector<int64_t> &getAxes() const;

  // Keep the reduced dimensions or not. A value of `true` means this op will
  // preserve the rank of the output tensor.
  bool getKeepDims() const;

private:
  std::vector<int64_t> backward_shape;
  std::vector<int64_t> axes;
  int64_t keepdims;
};

class ReduceSumGradOp : public Op {
public:
  ReduceSumGradOp(ReduceSumOp *fwdOp,
                  const std::vector<int64_t> &backward_shape);
  std::unique_ptr<Op> clone() const override final;
  void setup() override final;

  const std::vector<GradInOutMapper> &gradInputInfo() const override final;
  const std::map<int, int> &gradOutToNonGradIn() const override final;
  const std::vector<int64_t> &backwardShape() const;

private:
  TensorInfo outputTensorInfo;
  const std::vector<int64_t> backward_shape;
};

} // namespace willow

#endif
