// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATORSCALEOP_HPP
#define GUARD_NEURALNET_ACCUMULATORSCALEOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

// Update accumulator by scaling by a factor
class AccumulatorScaleOp : public VarUpdateOp {
  OptimizerValue factor;

public:
  AccumulatorScaleOp(const OptimizerValue factor_, const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;
  static InIndex getFactorInIndex() { return 2; }
  const OptimizerValue &getFactor() const { return factor; }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  view::Regions modifies(InIndex) const override;
};

} // namespace popart

#endif
