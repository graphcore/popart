// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOSSSCALEUPDATE_HPP
#define GUARD_NEURALNET_LOSSSCALEUPDATE_HPP

#include <popart/op.hpp>

namespace popart {

// This op takes as inputs:
// - The loss scale update factor, used to update the loss scale tensor and
//   inverse loss scale tensors.
// - Any number > 0 of 'gradient statistics' tensors, each of which is 1D
//   tensor with 2 elements
//
// and outputs:
// - The loss scale update factor - a scalar tensor. This can be used to
//   inplace-modify the loss scale and inverse loss scale optimizer tensors
//   in the automatic loss scaling Transform. This can be optionally clipped to
//   the largest power of 2 that fits in fp16 (32768) by setting clipOutput to
//   true.
class LossScaleUpdateOp : public Op {
public:
  LossScaleUpdateOp(const OperatorIdentifier &_opid,
                    const DataType &updateFactorDType_,
                    bool clipOutput_,
                    const Op::Settings &settings_)
      : Op(_opid, settings_), updateFactorDType(updateFactorDType_),
        clipOutput(clipOutput_) {}

  void setup() final;

  // The loss scale update factor
  static InIndex getLossScaleUpdateFactorInIndex() { return 0; }

  // Gradient tensor statistics are inputs at indices 0-N
  static InIndex getFirstStatisticsTensorInIndex() { return 2; }

  static InIndex getLossScalingInIndex() { return 1; }

  // The factor by which to multiply the loss scale tensor
  static OutIndex getUpdatedLossScaleUpdateFactorOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::unique_ptr<Op> clone() const override;

  DataType getUpdateFactorDType() const { return updateFactorDType; }
  bool getClipOutput() const { return clipOutput; }

  // This Op aliases and modifies the input at index
  // getLossScaleUpdateFactorInIndex()
  view::Regions aliases(InIndex in, OutIndex) const override;
  view::Regions modifies(InIndex) const override;
  void growAliasModel(AliasModel &m) const override;

private:
  DataType updateFactorDType;
  bool clipOutput;
};

} // namespace popart

#endif
