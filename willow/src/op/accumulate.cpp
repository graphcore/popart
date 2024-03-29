// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <popart/op/accumulate.hpp>
#include <popart/opserialiser.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
struct OperatorIdentifier;

std::ostream &operator<<(std::ostream &os, const AccumulationType &at) {
  switch (at) {
  case AccumulationType::Add:
    return os << "AccumulationType::Add";
  case AccumulationType::DampenedAdd:
    return os << "AccumulationType::DampenedAdd";
  case AccumulationType::DampenedAddSquare:
    return os << "AccumulationType::DampenedAddSquare";
  case AccumulationType::DecayAdd:
    return os << "AccumulationType::DecayAdd";
  case AccumulationType::DecayAddSquare:
    return os << "AccumulationType::DecayAddSquare";
  case AccumulationType::MovingAverage:
    return os << "AccumulationType::MovingAverage";
  case AccumulationType::MovingAverageSquare:
    return os << "AccumulationType::MovingAverageSquare";
  case AccumulationType::Infinity:
    return os << "AccumulationType::Infinity";
  case AccumulationType::Mean:
    return os << "AccumulationType::Mean";
  default:
    throw internal_error(
        "Unhandled AccumulationType with int value {} in operator<<.",
        static_cast<int>(at));
  }
}

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
  // TODO(T43862): Support DampenedAddSquareg in SparseAccumulatex
  return type == AccumulationType::Add || type == AccumulationType::DampenedAdd;
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
