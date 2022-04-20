// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/dropoutbase.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/op/randombase.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

DropoutBaseOp::DropoutBaseOp(const OperatorIdentifier &opid_,
                             float ratio_,
                             const Op::Settings &settings_)
    : RandomBaseOp(opid_, OptionalDataType(), settings_), ratio(ratio_) {}

std::unique_ptr<Op> DropoutBaseOp::clone() const {
  return std::make_unique<DropoutBaseOp>(*this);
}

// Dropout in testing mode can be replaced by the identity
bool DropoutBaseOp::canBeReplacedByIdentity() const {
  return (getIr().isTesting() || ratio == 0);
}

void DropoutBaseOp::configureShardedOp(Op *const shardedOp,
                                       const Settings *const settings_) const {
  Op::configureShardedOp(shardedOp, settings_);
  // TODO: T29800
}

float DropoutBaseOp::validateRatioAttribute(const OpCreatorInfo &info) {
  float ratio = info.attributes.getAttribute<Attributes::Float>("ratio", 0.5f);
  // If invalid probability for ratio supplied, throw error.
  if (ratio < float(0.) || ratio >= float(1.)) {
    throw error("{} ratio value {} is not valid. Please use a value in the "
                "interval [0,1)",
                info.opid,
                ratio);
  }
  return ratio;
}

} // namespace popart
