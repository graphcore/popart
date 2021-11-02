// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATORSCALEOP_HPP
#define GUARD_NEURALNET_ACCUMULATORSCALEOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

/**
 * @brief Inplace multiplies a tensor by an OptimizerValue factor.
 *
 * As with other Ops that consume OptimizerValues, will only have an input
 * tensor for the value if the OptimizerValue is not const.
 *
 * Will directly zero the input tensor if the factor is const and 0.
 */
class AccumulatorScaleOp : public VarUpdateOp {
  OptimizerValue factor;

public:
  AccumulatorScaleOp(const OptimizerValue factor_, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  std::map<InIndex, TensorId> optimizerInputs() const override;
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  static InIndex getFactorInIndex() { return 2; }
  const OptimizerValue &getFactor() const { return factor; }
  float getSubgraphValue() const override { return getLowSubgraphValue(); }
  view::Regions modifies(InIndex) const override;
};

} // namespace popart

#endif
