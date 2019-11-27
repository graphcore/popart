#ifndef GUARD_NEURALNET_CLIP_HPP
#define GUARD_NEURALNET_CLIP_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class OpSerialiserBase;

class ClipOp : public ElementWiseUnaryOp {
public:
  ClipOp(const OperatorIdentifier &_opid,
         float min_,
         float max_,
         const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setClipMin(float value) { min = value; }
  float getClipMin() const;
  void setClipMax(float value) { max = value; }
  float getClipMax() const;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
  bool canBeReplacedByIdentity() override;

private:
  float min;
  float max;
};

class ClipInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  ClipInplaceOp(const ClipOp &);
  std::unique_ptr<Op> clone() const final;

  float getClipMin() const;
  float getClipMax() const;
  void appendOutlineAttributes(OpSerialiserBase &) const override;

private:
  float min;
  float max;
};

class ClipGradOp : public ClipOp {
public:
  ClipGradOp(const ClipOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  // The index at which the output of the Clip (the "clipped" tensor)
  // is an input to this ClipGradOp
  static InIndex getClippedInIndex() { return 1; }

  // The index at which the gradient of the output of
  // the Clip is an input to this ClipGradOp
  static InIndex getGradClippedInIndex() { return 0; }
};

} // namespace popart

#endif
