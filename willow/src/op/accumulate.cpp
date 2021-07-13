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

AccumulateBaseOp::AccumulateBaseOp(const OperatorIdentifier &opid,
                                   AccumulationType type_,
                                   OptimizerValue factor_,
                                   const Op::Settings &opSettings)
    : VarUpdateWithUpdaterOp(opid, opSettings), type(type_), factor(factor_) {}

std::unique_ptr<Op> AccumulateOp::clone() const {
  return std::make_unique<AccumulateOp>(*this);
}

AccumulateOp::AccumulateOp(AccumulationType type,
                           OptimizerValue factor,
                           const Op::Settings &opSettings)
    : AccumulateBaseOp(Onnx::CustomOperators::Accumulate,
                       type,
                       factor,
                       opSettings) {}

std::unique_ptr<Op> RescaleAccumulateOp::clone() const {
  return std::make_unique<RescaleAccumulateOp>(*this);
}

std::map<InIndex, TensorId> RescaleAccumulateOp::optimizerInputs() const {
  auto m = AccumulateBaseOp::optimizerInputs();
  m.insert({getRescaleRatioInIndex(), inId(getRescaleRatioInIndex())});
  return m;
}

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

} // namespace popart
