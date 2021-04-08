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
            RandomReferenceId referenceId_,
            bool outputMask_,
            const Op::Settings &settings_);

public:
  DropoutOp(const OperatorIdentifier &_opid,
            float ratio_,
            const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() override;

  void appendAttributes(OpSerialiserBase &os) const override;

  // Additional mask output
  static OutIndex getMaskOutIndex() { return 1; }

  void setOutputMask(bool v) { outputMask = v; }
  bool getOutputMask() const { return outputMask; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setReferenceId(RandomReferenceId id) { referenceId = id; }
  RandomReferenceId getReferenceId() const { return referenceId; }

  TensorId getReferenceTensorId();

protected:
  RandomReferenceId referenceId;

private:
  bool outputMask = false;
};

class DropoutGradOp : public DropoutOp {
public:
  DropoutGradOp(const DropoutOp &fwdOp);

  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const override;
  const std::map<int, int> &gradOutToNonGradIn() const override;

  static InIndex getGradInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace popart

#endif
