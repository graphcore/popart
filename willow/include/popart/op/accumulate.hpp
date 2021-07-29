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

  std::unique_ptr<Op> clone() const override = 0;

  std::map<InIndex, TensorId> optimizerInputs() const override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  static constexpr InIndex getFactorInIndex() { return 2; }

  float getSubgraphValue() const final {
    if (type == AccumulationType::MovingAverage ||
        type == AccumulationType::MovingAverageSquare ||
        type == AccumulationType::Infinity) {
      return getHighSubgraphValue();
    } else {
      return getLowSubgraphValue();
    }
  }

  AccumulationType getAccumulationType() const { return type; }
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

class RescaleAccumulateOp final : public AccumulateBaseOp {
public:
  RescaleAccumulateOp(AccumulationType type_,
                      OptimizerValue factor_,
                      const Op::Settings &);

  std::unique_ptr<Op> clone() const final;

  std::map<InIndex, TensorId> optimizerInputs() const final;

  static InIndex getRescaleRatioInIndex() { return 3; }
};

/**
 *
 * Say you have:
 * w -> Gather -> x
 *
 * In backward pass you have:
 * dW <- GatherGrad <- x
 *
 * and when the optimiser step is grown:
 * dW <- GatherGrad <- x
 *  \
 *   Accumulate -> accum'
 *  /
 * accum
 *
 * GatherGrad is essentially a scatter. Then we Accumulate the resultant dW on
 * accum. This involves creating an extra dW tensor, so instead we can do:
 *
 *               x
 *               |
 *               V
 * accum -> SparseAccumulate -> accum'
 *
 * Where SparseAccumulate can in one operation, without extra space, accumulate
 * the slices of x into accum as required.
 *
 * ---------
 *
 * The input tensor at getOriginalVarToUpdateInIndex() is an optional input.
 * This is can be used when two different views of the weight are consumed in
 * the forward pass (by ops that will be autodiffed), and one of those ops is a
 * Gather, thus requiring a SparseAccumulate in the weight update step.
 *
 * We connect getOriginalVarToUpdateInIndex() to the other view of the weight
 * than the one this SparseAccumulate is for. Then, SparseAccumulateOpx will
 * clone that tensor (and its layout) when creating accum.
 * \sa SparseAccumulateOpx::createInputTensor for further motivation of why it
 * does this.
 *
 * You probably do not need this outside of the TiedGatherPattern.
 */
class SparseAccumulateOp final : public AccumulateBaseOp {
public:
  SparseAccumulateOp(AccumulationType type,
                     const OptimizerValue &factor,
                     unsigned axis,
                     const Op::Settings &);

  std::unique_ptr<Op> clone() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  static constexpr InIndex getIndicesInIndex() { return 3; }
  static constexpr InIndex getOriginalVarToUpdateInIndex() { return 4; }

  std::set<InIndex> optionalInputs() const override {
    return {getOriginalVarToUpdateInIndex()};
  }

  unsigned getAxis() const;

  static bool supportsAccumulationType(AccumulationType type);

private:
  unsigned axis;
};

} // namespace popart

#endif
