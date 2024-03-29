// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SGD1COMBO_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SGD1COMBO_HPP_

#include <memory>
#include <popart/op/sgdcombobase.hpp>
#include <popart/optimizervalue.hpp>

#include "popart/op.hpp"

namespace popart {

enum class OptimizerReductionType;

/**
 * \brief A single Op that encapsulates all the information needed to describe
 * an SGD1 optimiser step.
 *
 * The "1" in the name signifies that only one extra optimiser tensor (the accl
 * tensor) is required. \sa SGD for the definition of what SGD1 is.
 *
 * The "Combo" in the name signifies that this Op will later be decomposed into
 * many Ops and Tensors that actually implement the optimiser step. In this
 * case, by the SGD1Decompose pattern. \sa SGD1Decompose for the definition of
 * this decomposition.
 */
class SGD1ComboOp final : public SGDMComboBaseOp {
public:
  SGD1ComboOp(OptimizerValue initialSmm1,
              OptimizerValue initialDpsf1,
              OptimizerValue initialSwd1,
              OptimizerValue initialSlr1,
              OptimizerReductionType reductionType_,
              const Op::Settings &);

  SGD1ComboOp(OptimizerValue initialSmm1,
              OptimizerValue initialDpsf1,
              OptimizerValue initialSwd1,
              OptimizerValue initialSlr1,
              OptimizerValue initialMm,
              OptimizerValue initialWd,
              OptimizerValue initialNgsf1,
              OptimizerValue initialNdsf1,
              OptimizerReductionType reductionType_,
              const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SGD1COMBO_HPP_
