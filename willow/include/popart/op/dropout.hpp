// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DROPOUT_HPP
#define GUARD_NEURALNET_DROPOUT_HPP

#include <popart/op.hpp>

namespace popart {

class DropoutOp : public Op {
public:
  DropoutOp(const OperatorIdentifier &opid_,
            float ratio_,
            uint32_t seedModifier_,
            bool outputMask_,
            const Op::Settings &settings_);

  DropoutOp(const OperatorIdentifier &_opid,
            float ratio_,
            const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Inputs
  static InIndex getInIndex() { return 0; }

  // Ouputs
  static OutIndex getOutIndex() { return 0; }
  static OutIndex getMaskOutIndex() { return 1; }

  bool canBeReplacedByIdentity() override;

  uint32_t getSeedModifier() const;
  void setSeedModifier(uint32_t sm);

  float getRatio() const;
  void setRatio(float r);

  void setOutputMask(bool v) { output_mask = v; }
  bool getOutputMask() const { return output_mask; }

  bool requiresRandomSeed() const override { return true; }
  InIndex getSeedInIndex() const override { return 1; }

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  float getSubgraphValue() const override;

  bool canShard() const override { return true; }

  std::map<TensorId, std::vector<TensorId>>
  shard(const std::map<TensorId, std::vector<TensorId>> &inputs) override;

  const TensorId &getReferenceTensorId();

protected:
  float ratio;
  uint32_t seedModifier;
  OpId partnerId;

private:
  TensorId refTensorId;
  bool output_mask = false;
};

class DropoutGradOp : public DropoutOp {
public:
  DropoutGradOp(const DropoutOp &fwdOp);

  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
