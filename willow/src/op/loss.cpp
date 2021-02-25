// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <sstream>
#include <popart/error.hpp>
#include <popart/ir.hpp>
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

LossOp::LossOp(const OperatorIdentifier &_opid,
               const Op::Settings &settings_,
               const ReductionType reduction_)
    : Op(_opid, settings_), reduction_type_(reduction_) {}

bool LossOp::isLossOp() const { return true; }

ScaleByReplication
LossOp::getScaleByReplication(ReductionType reduction) const {
  // We are only interested in grad op behaviour when doing training
  if (!getIr().isTraining()) {
    return ScaleByReplication::No;
  }

  auto opts = getIr().getSessionOptions();

  if (opts.getGlobalReplicationFactor() > 1 &&
      reduction == ReductionType::Mean &&
      opts.accumulationAndReplicationReductionType ==
          ReductionType::NoReduction) {
    logging::warn(
        "Loss Op, {}, has 'mean' reduction type, and graph replication is "
        "enabled. In the corresponding backwards op, the output gradient will "
        "be scaled by the inverse of the replication factor so that the "
        "gradient reduction computes the mean. This behaviour has been "
        "deprecated and will be removed in a future release. Instead, please "
        "set the SessionOption 'accumulationAndReplicationReductionType' to "
        "ReductionType::Mean for the equivalent behaviour.",
        str());
    return ScaleByReplication::Yes;
  } else {
    return ScaleByReplication::No;
  }
}

} // namespace popart
