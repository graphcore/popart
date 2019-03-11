#ifndef GUARD_NEURALNET_SCALE_HPP
#define GUARD_NEURALNET_SCALE_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class OpSerialiserBase;

// y = scale_factor * x
class ScaleOp : public ElementWiseUnaryOp {
public:
  ScaleOp(const OperatorIdentifier &_opid,
          float scale_,
          const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setScaleFactor(float value) { scale_factor = value; }
  float getScaleFactor() const;
  void appendAttributes(OpSerialiserBase &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
  bool canBeReplacedByIdentity() override;

private:
  float scale_factor;
};

// TODO: unify inplace elementwise op class logic (T6801)
class ScaleInplaceOp : public Op {
public:
  ScaleInplaceOp(const ScaleOp &);
  void setup() final;
  // This in-place Op modifies its unique input at InIndex 0

  view::Region modifies(InIndex index) const final { return uses(index); }
  view::Region aliases(InIndex index) const final { return uses(index); }
  // "uses" is still the full region
  // "fwdRegMap" and "bwdRegMap" are still the identity

  // TODO T6801 : don't repeat scale_factor
  float getScaleFactor() const;
  void appendAttributes(OpSerialiserBase &) const override;

private:
  float scale_factor;
};

class ScaleGradOp : public ScaleOp {
public:
  ScaleGradOp(const ScaleOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
