#ifndef GUARD_NEURALNET_REDUCESUM_HPP
#define GUARD_NEURALNET_REDUCESUM_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class ReduceSumOp : public Op {
public:
  ReduceSumOp(const OpConstructorBundle &);
  ReduceSumOp(const OpConstructorBundle &,
              const std::vector<int64_t> &axes,
              int64_t keepdims);
  ReduceSumOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() override;

  // A list of integers, along which to reduce. These axes will either be
  // removed or have size 1, depending on the value of getKeepDims.
  const std::vector<int64_t> &getAxes() const;

  // Keep the reduced dimensions or not. A value of `true` means this op will
  // preserve the rank of the input tensor, inserting 1 at reduced axes
  bool getKeepDims() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  // The input shape, with '1' inserted in reduction axes.
  // This is the same as the output shape if keepdims is true.
  Shape backward_shape;
  std::vector<int64_t> axes;
  int64_t keepdims;
};

class ReduceSumGradOp : public Op {
public:
  ReduceSumGradOp(ReduceSumOp *fwdOp, const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  const Shape &backwardShape() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  TensorInfo outputTensorInfo;
  // Copied from constructing ReduceSumOp. In this context, it is
  // the shape of this grad Op's input, but with '1's inserted where
  // broadcasts are required to obtain the gradient of the fwd Op's input
  const Shape backward_shape;
};

} // namespace poponnx

#endif
