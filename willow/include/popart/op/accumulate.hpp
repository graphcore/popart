// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATEOP_HPP
#define GUARD_NEURALNET_ACCUMULATEOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

enum class AccumulationType {
  Add = 0,            // accum += g
  DampenedAdd,        // accum += f * g
  MovingAverage,      // accum = f * accum + (1 - f) * g
  MovingAverageSquare // accum = f * accum + (1 - f) * g^2
};

class AccumulateOp : public VarUpdateWithUpdaterOp {

public:
  AccumulateOp(const TensorId &varToUpdate,
               AccumulationType type_,
               OptimizerValue factor_,
               const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::unique_ptr<Op> cloneWithNewName(const TensorId &newName) const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;
  static InIndex getFactorInIndex() { return 2; }
  float getSubgraphValue() const final {
    if (type == AccumulationType::MovingAverage ||
        type == AccumulationType::MovingAverageSquare) {
      return getHighSubgraphValue();
    } else {
      return getLowSubgraphValue();
    }
  }

  const AccumulationType &getAccumulationType() const { return type; }
  const OptimizerValue &getFactor() const { return factor; }

private:
  AccumulationType type;
  const OptimizerValue factor;
};

} // namespace popart

#endif
