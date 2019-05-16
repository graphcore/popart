#ifndef GUARD_NEURALNET_FLATTEN_HPP
#define GUARD_NEURALNET_FLATTEN_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/reshape.hpp>

namespace poponnx {

class FlattenBaseOp : public Op {
public:
  FlattenBaseOp(const OperatorIdentifier &_opid,
                int64_t axis_,
                const Op::Settings &settings_);

  void setup() final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setAxis(int64_t value);
  int64_t getAxis() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void appendAttributes(OpSerialiserBase &) const override;

  // currently these are conservative TODO T6973
  view::RegMap fwdRegMap(InIndex) const final;
  view::RegMap bwdRegMap(InIndex) const final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  int64_t axis;
};

// Corresponds to the ONNX Flatten op for N-dimensional tensors.
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#flatten
class FlattenOp : public FlattenBaseOp {
public:
  FlattenOp(const OperatorIdentifier &_opid,
            int64_t axis_,
            const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {{Onnx::CustomOperators::FlattenInplace, 10}};
  }
};

class FlattenInplaceOp : public FlattenBaseOp {
public:
  FlattenInplaceOp(const OperatorIdentifier &_opid,
                   int64_t axis_,
                   const Op::Settings &settings_);
  FlattenInplaceOp(const FlattenOp &);
  std::unique_ptr<Op> clone() const final;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  // modifies and uses are still the defaults, but aliases changes
  // to be the same as uses (the full out region)
  view::Region aliases(InIndex index) const final;
};

// The gradient of a flatten is a reshape back to the original shape
class FlattenGradOp : public ReshapeOp {
public:
  FlattenGradOp(const FlattenBaseOp &fwdOp);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace poponnx

#endif
