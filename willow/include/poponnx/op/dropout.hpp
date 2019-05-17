#ifndef GUARD_NEURALNET_DROPOUT_HPP
#define GUARD_NEURALNET_DROPOUT_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class DropoutOp : public Op {
public:
  DropoutOp(const OperatorIdentifier &_opid,
            float ratio_,
            const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  void setup() override;

  void appendAttributes(OpSerialiserBase &) const override;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  bool canBeReplacedByIdentity() override;

  uint32_t getSeedModifier() const { return seedModifier; }
  void setSeedModifier(uint32_t sm) { seedModifier = sm; }
  float getRatio() const { return ratio; }
  void setRatio(float r) { ratio = r; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  float ratio;
  uint32_t seedModifier;
};

class DropoutGradOp : public Op {
public:
  DropoutGradOp(const DropoutOp &fwdOp);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getGradInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
  uint32_t getSeedModifier() const { return seedModifier; }
  float getRatio() const { return ratio; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  float ratio;
  uint32_t seedModifier;
};

} // namespace poponnx

#endif
