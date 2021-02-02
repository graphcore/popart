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
  // Gradient tensor statistics are inputs at indices 1-N

  static OutIndex getUpdatedLossScaleOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::unique_ptr<Op> clone() const override;

  // This Op aliases and modifies the input at index getLossScaleInIndex()
  view::Regions aliases(InIndex in, OutIndex) const override;
  view::Regions modifies(InIndex) const override;
};

} // namespace popart

#endif
