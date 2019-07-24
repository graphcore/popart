#ifndef GUARD_NEURALNET_DROPOUT_HPP
#define GUARD_NEURALNET_DROPOUT_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class DropoutBaseOp : public Op {
public:
  DropoutBaseOp(const OperatorIdentifier &opid_,
                float ratio_,
                uint32_t seedModifier_,
                const Op::Settings &settings_);

  uint32_t getSeedModifier() const;
  void setSeedModifier(uint32_t sm);
  float getRatio() const;
  void setRatio(float r);

  float getSubgraphValue() const final;

protected:
  float ratio;
  uint32_t seedModifier;
};

class DropoutOp : public DropoutBaseOp {
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
  static OutIndex getSeedOutIndex() { return 2; }

  bool canBeReplacedByIdentity() override;

  bool returnMask() const { return output_mask; }

private:
  bool output_mask = false;
};

class DropoutGradOp : public DropoutBaseOp {
public:
  DropoutGradOp(const DropoutOp &fwdOp);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getSeedInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
