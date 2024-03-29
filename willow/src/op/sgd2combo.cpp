// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <utility>
#include <popart/op/sgd2combo.hpp>
#include <popart/optimizer.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/op/sgdcombobase.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

SGD2ComboOp::SGD2ComboOp(OptimizerValue initialSmm1,
                         OptimizerValue initialDpsf1,
                         OptimizerValue initialSwd1,
                         OptimizerValue initialSlr1,
                         bool withGradAccum_,
                         OptimizerReductionType reductionType_,
                         DataType accumType_,
                         DataType accl1Type_,
                         const Op::Settings &settings_)
    : SGDMComboBaseOp(Onnx::CustomOperators::SGD2Combo,
                      std::move(initialSmm1),
                      std::move(initialDpsf1),
                      std::move(initialSwd1),
                      std::move(initialSlr1),
                      reductionType_,
                      settings_),
      withGradAccum(withGradAccum_), accumType(accumType_),
      accl1Type(accl1Type_) {
  if (reductionType == OptimizerReductionType::AcclReduce) {
    throw error("SGD2 does not support OptimizerReductionType::AcclReduce, "
                "because the accl tensor cannot be reduced across replicas "
                "(without introducing rf scaling).");
  }
  if (!withGradAccum && reductionType == OptimizerReductionType::AccumReduce) {
    throw error("SGD2 cannot perform OptimizerReductionType::AccumReduce when "
                "gradient accumulation is not enabled, as there will be no "
                "accum tensor to reduce.");
  }

  const auto validateIsFloatingPoint = [](const DataType dt,
                                          const std::string &name) {
    if (!((dt == DataType::FLOAT) || (dt == DataType::FLOAT16))) {
      throw error("SGD2 only supports FLOAT or FLOAT16 for DataType of {}. You "
                  "passed {}.",
                  name,
                  dt);
    }
  };
  validateIsFloatingPoint(accumType, "accum");
  validateIsFloatingPoint(accl1Type, "accl1");
}

SGD2ComboOp::SGD2ComboOp(OptimizerValue initialSmm1,
                         OptimizerValue initialDpsf1,
                         OptimizerValue initialSwd1,
                         OptimizerValue initialSlr1,
                         OptimizerValue initialMm,
                         OptimizerValue initialWd,
                         OptimizerValue initialNgsf2,
                         OptimizerValue initialNdsf2,
                         bool withGradAccum_,
                         OptimizerReductionType reductionType_,
                         DataType accumType_,
                         DataType accl1Type_,
                         const Op::Settings &settings_)
    : SGDMComboBaseOp(Onnx::CustomOperators::SGD2Combo,
                      std::move(initialSmm1),
                      std::move(initialDpsf1),
                      std::move(initialSwd1),
                      std::move(initialSlr1),
                      std::move(initialMm),
                      std::move(initialWd),
                      std::move(initialNgsf2),
                      std::move(initialNdsf2),
                      reductionType_,
                      settings_),
      withGradAccum(withGradAccum_), accumType(accumType_),
      accl1Type(accl1Type_) {
  if (reductionType == OptimizerReductionType::AcclReduce) {
    throw error("SGD2 does not support OptimizerReductionType::AcclReduce, "
                "because the accl tensor cannot be reduced across replicas "
                "(without introducing rf scaling).");
  }
  if (!withGradAccum && reductionType == OptimizerReductionType::AccumReduce) {
    throw error("SGD2 cannot perform OptimizerReductionType::AccumReduce when "
                "gradient accumulation is not enabled, as there will be no "
                "accum tensor to reduce.");
  }

  const auto validateIsFloatingPoint = [](const DataType dt,
                                          const std::string &name) {
    if (!((dt == DataType::FLOAT) || (dt == DataType::FLOAT16))) {
      throw error("SGD2 only supports FLOAT or FLOAT16 for DataType of {}. You "
                  "passed {}.",
                  name,
                  dt);
    }
  };
  validateIsFloatingPoint(accumType, "accum");
  validateIsFloatingPoint(accl1Type, "accl1");
}

std::unique_ptr<Op> SGD2ComboOp::clone() const {
  return std::make_unique<SGD2ComboOp>(*this);
}

} // namespace popart
