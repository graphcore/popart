// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOSSSCALEUPDATE_HPP
#define GUARD_NEURALNET_LOSSSCALEUPDATE_HPP

#include <popart/op.hpp>

namespace popart {

// This op takes inputs:
// - the loss scale scalar tensor
// - any number > 0 of 'gradient statistics' tensors, each of which is 1D
//   tensor with 2 elements

class LossScaleUpdateOp : public Op {
public:
  LossScaleUpdateOp(const OperatorIdentifier &_opid,
                    const Op::Settings &settings_)
      : Op(_opid, settings_) {}

  void setup() final;

  static InIndex getLossScaleInIndex() { return 0; }

  // This is the compound scalar tensor that has been divided by the loss
  // scale factor. Depending on the optimizer chosen by the user, this could
  // be one of 'scaledLearningRate0' or 'dampeningScaleFactor1'. See
  // optimizer.hpp for details.
  static InIndex getInverseLossScaleInIndex() { return 1; }

  // Gradient tensor statistics are inputs at indices 2-N
  static InIndex getFirstStatisticsTensorInIndex() { return 2; }

  static OutIndex getUpdatedLossScaleOutIndex() { return 0; }
  static OutIndex getUpdatedInverseLossScaleOutIndex() { return 1; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::unique_ptr<Op> clone() const override;

  // This Op aliases and modifies the input at index getLossScaleInIndex()
  view::Regions aliases(InIndex in, OutIndex) const override;
  view::Regions modifies(InIndex) const override;
};

} // namespace popart

#endif
