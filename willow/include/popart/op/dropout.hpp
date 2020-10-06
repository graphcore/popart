// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DROPOUT_HPP
#define GUARD_NEURALNET_DROPOUT_HPP

#include <popart/op/dropoutbase.hpp>

namespace popart {

class DropoutOp : public DropoutBaseOp {
protected:
  // protected constructor used in grad op
  DropoutOp(const OperatorIdentifier &opid_,
            float ratio_,
            uint32_t seedModifier_,
            bool outputMask_,
            const Op::Settings &settings_);

public:
  DropoutOp(const OperatorIdentifier &_opid,
            float ratio_,
            const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  // Additional mask output
  static OutIndex getMaskOutIndex() { return 1; }

  void setOutputMask(bool v) { outputMask = v; }
  bool getOutputMask() const { return outputMask; }

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  const TensorId &getReferenceTensorId();

  // TODO (T25465): this setter can be removed once dropout is outlineable
  void setSeedModifier(uint32_t sm) { seedModifier = sm; }

protected:
  OpId partnerId;

private:
  TensorId refTensorId;
  bool outputMask = false;
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
