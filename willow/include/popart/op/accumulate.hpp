// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATEOP_HPP
#define GUARD_NEURALNET_ACCUMULATEOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

enum class AccumulationType {
  Add = 0,             // accum += g
  DampenedAdd,         // accum += f * g
  DampenedAddSquare,   // accum += f * g^2
  DecayAdd,            // accum = f * accum + g
  DecayAddSquare,      // accum = f * accum + g^2
  MovingAverage,       // accum = f * accum + (1 - f) * g
  MovingAverageSquare, // accum = f * accum + (1 - f) * g^2
  Infinity             // accum = max(f * accum, abs(g))
};

class AccumulateOp : public VarUpdateWithUpdaterOp {

public:
  AccumulateOp(AccumulationType type_,
               OptimizerValue factor_,
               const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;
  static InIndex getFactorInIndex() { return 2; }
  float getSubgraphValue() const final {
    if (type == AccumulationType::MovingAverage ||
        type == AccumulationType::MovingAverageSquare ||
        type == AccumulationType::Infinity) {
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
