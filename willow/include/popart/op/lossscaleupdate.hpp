// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOSSSCALEUPDATE_HPP
#define GUARD_NEURALNET_LOSSSCALEUPDATE_HPP

#include <popart/op.hpp>

namespace popart {

// This op takes as inputs:
// - any number > 0 of 'gradient statistics' tensors, each of which is 1D
//   tensor with 2 elements
//
// and outputs:
// - The loss scale update factor - a scalar tensor. This can be used to
//   inplace-modify the loss scale and inverse loss scale optimizer tensors
//   in the automatic loss scaling Transform
class LossScaleUpdateOp : public Op {
public:
  LossScaleUpdateOp(const OperatorIdentifier &_opid,
                    const DataType &updateFactorDType_,
                    const Op::Settings &settings_)
      : Op(_opid, settings_), updateFactorDType(updateFactorDType_) {}

  void setup() final;

  // Gradient tensor statistics are inputs at indices 0-N
  static InIndex getFirstStatisticsTensorInIndex() { return 0; }

  // The factor by which to multiply the loss scale tensor
  static OutIndex getLossScaleUpdateFactorOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::unique_ptr<Op> clone() const override;

  DataType getUpdateFactorDType() const { return updateFactorDType; }

private:
  DataType updateFactorDType;
};

} // namespace popart

#endif
