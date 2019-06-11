#ifndef GUARD_NEURALNET_REDUCE_HPP
#define GUARD_NEURALNET_REDUCE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class ReduceOp : public Op {
public:
  ReduceOp(const OperatorIdentifier &_opid,
           const std::vector<int64_t> &axes,
           const int64_t keepdims,
           const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  // A list of integers, along which to reduce. These axes will either be
  // removed or have size 1, depending on the value of getKeepDims.
  const std::vector<int64_t> &getAxes() const;

  // Keep the reduced dimensions or not. A value of `true` means this op will
  // preserve the rank of the input tensor, inserting 1 at reduced axes
  bool getKeepDims() const;

  void setAxes(std::vector<int64_t> value);
  void setKeepDims(int64_t value);

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void appendAttributes(OpSerialiserBase &) const override;

  bool canBeReplacedByIdentity() override;

  bool isOutlineable() const final { return false; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  const Shape &backwardShape() const;

protected:
  // The input shape, with '1' inserted in reduction axes.
  // This is the same as the output shape if keepdims is true.
  Shape backward_shape;
  std::vector<int64_t> axes;
  int64_t keepdims;
};

class ReduceGradOp : public Op {
public:
  ReduceGradOp(const AiGraphcoreOpIdV1 &opid,
               const ReduceOp &fwdOp,
               const Shape &backward_shape);
  std::unique_ptr<Op> clone() const override;
  void setup() override;

  // A list of integers, along which have been reduced.
  const std::vector<int64_t> &getAxes() const;

  const std::vector<GradInOutMapper> &gradInputInfo() const override;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  const Shape &backwardShape() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

protected:
  TensorInfo outputTensorInfo;
  // Copied from constructing ReduceOp. In this context, it is
  // the shape of this grad Op's input, but with '1's inserted where
  // broadcasts are required to obtain the gradient of the fwd Op's input
  const Shape backward_shape;
  std::vector<int64_t> axes;
};

} // namespace poponnx

#endif
