// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

/**************** AccumulateBaseOp ****************/

AccumulateBaseOp::AccumulateBaseOp(const OperatorIdentifier &opid,
                                   AccumulationType type_,
                                   OptimizerValue factor_,
                                   const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(opid, opSettings), type(type_), factor(factor_) {}

std::map<InIndex, TensorId> AccumulateBaseOp::optimizerInputs() const {
  std::map<InIndex, TensorId> m;
  if (!factor.isConst()) {
    auto index = getFactorInIndex();
    m.insert({index, inId(index)});
  }
  return m;
}

void AccumulateBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);

  os.appendAttribute("type", static_cast<int>(getAccumulationType()));

  if (factor.isConst()) {
    os.appendAttribute("const factor", getFactor().val());
  }
}

/**************** RescaleAccumulateOp ****************/

RescaleAccumulateOp::RescaleAccumulateOp(AccumulationType type,
                                         OptimizerValue factor,
                                         const Op::Settings &opSettings)
    : AccumulateBaseOp(Onnx::CustomOperators::RescaleAccumulate,
                       type,
                       factor,
                       opSettings) {
  switch (type) {
  case AccumulationType::MovingAverage:
  case AccumulationType::MovingAverageSquare:
  case AccumulationType::Infinity:
    break;
  default:
    throw error("Unsupported AccumulationType in RescaleAccumulateOp {}",
                static_cast<int>(type));
  }
}

std::map<InIndex, TensorId> RescaleAccumulateOp::optimizerInputs() const {
  auto m = AccumulateBaseOp::optimizerInputs();
  m.insert({getRescaleRatioInIndex(), inId(getRescaleRatioInIndex())});
  return m;
}

std::unique_ptr<Op> RescaleAccumulateOp::clone() const {
  return std::make_unique<RescaleAccumulateOp>(*this);
}

/**************** AccumulateOp ****************/

AccumulateOp::AccumulateOp(AccumulationType type,
                           OptimizerValue factor,
                           const Op::Settings &opSettings)
    : AccumulateBaseOp(Onnx::CustomOperators::Accumulate,
                       type,
                       factor,
                       opSettings) {}

std::unique_ptr<Op> AccumulateOp::clone() const {
  return std::make_unique<AccumulateOp>(*this);
}

/**************** SparseAccumulateOp ****************/

bool SparseAccumulateOp::supportsAccumulationType(const AccumulationType type) {
  return type == AccumulationType::Add ||
         type == AccumulationType::DampenedAdd ||
         type == AccumulationType::DampenedAddSquare;
}

SparseAccumulateOp::SparseAccumulateOp(const AccumulationType type_,
                                       const OptimizerValue &factor_,
                                       const unsigned axis_,
                                       const Op::Settings &settings_)
    : AccumulateBaseOp(Onnx::CustomOperators::SparseAccumulate,
                       type_,
                       factor_,
                       settings_),
      axis(axis_) {
  if (!supportsAccumulationType(type)) {
    throw error(
        "SparseAccumulateOp only supports AccumulationTypes: Add, DampenedAdd, "
        "and DampenedAddSquare. You passed AccumulationType with int value: {}",
        static_cast<int>(type));
  }
}

std::unique_ptr<Op> SparseAccumulateOp::clone() const {
  return std::make_unique<SparseAccumulateOp>(*this);
}

unsigned SparseAccumulateOp::getAxis() const { return axis; }

void SparseAccumulateOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  AccumulateBaseOp::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
}

} // namespace popart
