#ifndef GUARD_NEURALNET_SCALE_HPP
#define GUARD_NEURALNET_SCALE_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

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
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
  bool canBeReplacedByIdentity() override;

private:
  float scale_factor;
};

class ScaleInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  ScaleInplaceOp(const ScaleOp &);
  std::unique_ptr<Op> clone() const final;

  // TODO T6801 : don't repeat scale_factor
  float getScaleFactor() const;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

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

} // namespace popart

#endif
