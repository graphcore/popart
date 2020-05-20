// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <sstream>
#include <popart/error.hpp>
#include <popart/op/loss.hpp>

namespace popart {

std::string LossOp::reductionTypeToString(ReductionType reduction) {
  if (reduction == ReductionType::Sum) {
    return "Sum";
  } else if (reduction == ReductionType::Mean) {
    return "Mean";
  } else if (reduction == ReductionType::NoReduction) {
    return "None";
  } else {
    throw error("Unexpected ReductionType. Cannot convert to string");
  }
}

ReductionType LossOp::reductionTypeFromString(std::string reduction) {
  if (reduction == "Sum") {
    return ReductionType::Sum;
  } else if (reduction == "Mean") {
    return ReductionType::Mean;
  } else if (reduction == "None") {
    return ReductionType::NoReduction;
  } else {
    throw error(
        "Unexpected ReductionType string, {}. Cannot convert to ReductionType",
        reduction);
  }
}

LossOp::LossOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

LossOp::LossOp(const Op &op) : Op(op) {}

bool LossOp::isLossOp() const { return true; }

} // namespace popart
