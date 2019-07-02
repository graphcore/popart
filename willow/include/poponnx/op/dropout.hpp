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
  void setup() final;

  void appendAttributes(OpSerialiserBase &) const override;

  // Inputs
  static InIndex getInIndex() { return 0; }

  // Ouputs
  static OutIndex getOutIndex() { return 0; }
  static OutIndex getMaskOutIndex() { return 1; }

  bool canBeReplacedByIdentity() override;

  uint32_t getSeedModifier() const { return seedModifier; }
  void setSeedModifier(uint32_t sm) { seedModifier = sm; }
  float getRatio() const { return ratio; }
  void setRatio(float r) { ratio = r; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool returnMask() const { return output_mask; }

private:
  bool output_mask = false;
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
