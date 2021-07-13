// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ACCUMULATEOP_HPP
#define GUARD_NEURALNET_ACCUMULATEOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

#include <ostream>

namespace popart {

// accum = getVarToUpdateInIndex
// g     = getUpdaterInIndex
// f     = getFactor if isConst else getFactorInIndex
//
// With AccumulationType::Mean, f is a counter which should have the value
// equal to the number of previously accumulated values in accum.
enum class AccumulationType {
  Add = 0,             // accum += g
  DampenedAdd,         // accum += f * g
  DampenedAddSquare,   // accum += f * g^2
  DecayAdd,            // accum = f * accum + g
  DecayAddSquare,      // accum = f * accum + g^2
  MovingAverage,       // accum = f * accum + (1 - f) * g
  MovingAverageSquare, // accum = f * accum + (1 - f) * g^2
  Infinity,            // accum = max(f * accum, abs(g))
  Mean                 // accum = (f/(f+1)) * accum + (1/(f+1)) * g
};

std::ostream &operator<<(std::ostream &os, const AccumulationType &at);

class AccumulateBaseOp : public VarUpdateWithUpdaterOp {
public:
  AccumulateBaseOp(const OperatorIdentifier &opid,
                   AccumulationType type_,
                   OptimizerValue factor_,
                   const Op::Settings &);

  std::map<InIndex, TensorId> optimizerInputs() const override;
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

protected:
  AccumulationType type;
  const OptimizerValue factor;
};

class AccumulateOp : public AccumulateBaseOp {
public:
  AccumulateOp(AccumulationType type,
               OptimizerValue factor,
               const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
};

/// The same as AccumulateOp however it also includes a rescale factor
/// that allows for the accumulator to be rescaled at the same time.
//
// Only supports the following AccumulateTypes
//   MovingAverage,       accum = rescale * f * accum + (1 - f) * g
//   MovingAverageSquare, accum = rescale * f * accum + (1 - f) * g^2
//   Infinity             accum = max(rescale * f * accum, abs(g))

class RescaleAccumulateOp : public AccumulateBaseOp {
public:
  RescaleAccumulateOp(AccumulationType type_,
                      OptimizerValue factor_,
                      const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  static InIndex getRescaleRatioInIndex() { return 3; }
};

} // namespace popart

#endif
