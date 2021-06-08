// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD2VARUPDATECOMBOOP_HPP
#define GUARD_NEURALNET_SGD2VARUPDATECOMBOOP_HPP

#include <popart/op/sgdcombobase.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervaluemap.hpp>

namespace popart {

/**
 * \brief A single Op that encapsulates all the information needed to describe
 * an SGD2 optimiser step.
 *
 * The "2" in the name signifies that two extra optimiser tensors (the accum
 * and accl1 tensors) may be required. \sa SGD for the definition of what SGD2
 * is.
 *
 * The "Combo" in the name signifies that this Op will later be decomposed into
 * many Ops and Tensors that actually implement the optimiser step. In this
 * case, by the SGD2Decompose pattern. \sa SGD2Decompose for the definition of
 * this decomposition.
 */
class SGD2ComboOp final : public SGDComboBaseOp {
public:
  SGD2ComboOp(OptimizerValue initialSmm1,
              OptimizerValue initialDpsf1,
              OptimizerValue initialSwd1,
              OptimizerValue initialSlr1,
              bool withGradAccum_,
              OptimizerReductionType reductionType_,
              DataType accumType_,
              DataType accl1Type_,
              const Op::Settings &);

  // Gradient accumulation
  const bool withGradAccum;

  // Data type of accumulator and momentum
  const DataType accumType;
  const DataType accl1Type;

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
