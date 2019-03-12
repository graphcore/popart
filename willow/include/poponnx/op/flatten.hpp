#ifndef GUARD_NEURALNET_FLATTEN_HPP
#define GUARD_NEURALNET_FLATTEN_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/reshape.hpp>

namespace poponnx {

// An aliasing variant of FlattenOp
// This can't be created by the inplace pattern at the moment.
class FlattenAliasOp : public Op {
public:
  FlattenAliasOp(const OperatorIdentifier &_opid,
                 int64_t axis_,
                 const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;

  void setup() final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setAxis(int64_t value);
  int64_t getAxis() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void appendAttributes(OpSerialiserBase &) const override;

  view::Region aliases(InIndex index) const final { return uses(index); }

private:
  int64_t axis;
};

// Corresponds to the ONNX Flatten op for N-dimensional tensors.
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#flatten
//
// This derives from FlattenAliasOp because it's more convenient to reuse
// FlattenAliasOp's implementation plus a copy than the other way around.
class FlattenOp : public FlattenAliasOp {
public:
  using FlattenAliasOp::FlattenAliasOp;
  std::unique_ptr<Op> clone() const override;
};

// The gradient of a flatten is a reshape back to the original shape
class FlattenGradOp : public ReshapeOp {
public:
  FlattenGradOp(const FlattenAliasOp &fwdOp);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace poponnx

#endif
