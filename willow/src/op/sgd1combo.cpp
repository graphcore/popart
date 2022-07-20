// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <utility>
#include <popart/op/sgd1combo.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/op/sgdcombobase.hpp"
#include "popart/optimizervalue.hpp"

namespace popart {

SGD1ComboOp::SGD1ComboOp(OptimizerValue initialSmm1,
                         OptimizerValue initialDpsf1,
                         OptimizerValue initialSwd1,
                         OptimizerValue initialSlr1,
                         OptimizerReductionType reductionType_,
                         const Op::Settings &settings_)
    : SGDMComboBaseOp(Onnx::CustomOperators::SGD1Combo,
                      std::move(initialSmm1),
                      std::move(initialDpsf1),
                      std::move(initialSwd1),
                      std::move(initialSlr1),
                      reductionType_,
                      settings_) {}

SGD1ComboOp::SGD1ComboOp(OptimizerValue initialSmm1,
                         OptimizerValue initialDpsf1,
                         OptimizerValue initialSwd1,
                         OptimizerValue initialSlr1,
                         OptimizerValue initialMm,
                         OptimizerValue initialWd,
                         OptimizerValue initialNgsf1,
                         OptimizerValue initialNdsf1,
                         OptimizerReductionType reductionType_,
                         const Op::Settings &settings_)
    : SGDMComboBaseOp(Onnx::CustomOperators::SGD1Combo,
                      std::move(initialSmm1),
                      std::move(initialDpsf1),
                      std::move(initialSwd1),
                      std::move(initialSlr1),
                      std::move(initialMm),
                      std::move(initialWd),
                      std::move(initialNgsf1),
                      std::move(initialNdsf1),
                      reductionType_,
                      settings_) {}

std::unique_ptr<Op> SGD1ComboOp::clone() const {
  return std::make_unique<SGD1ComboOp>(*this);
}

} // namespace popart
