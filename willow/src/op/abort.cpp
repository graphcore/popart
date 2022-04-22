// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/abort.hpp>
#include <popart/opmanager.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

AbortOp::AbortOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(Onnx::CustomOperators::Abort, settings_) {}

std::unique_ptr<Op> AbortOp::clone() const {
  return std::make_unique<AbortOp>(*this);
}

void AbortOp::setup() {
  if (hasInput(AbortOp::getInIndex())) {
    const auto &info = inInfo(AbortOp::getInIndex());
    if (info.nelms() > 1) {
      throw error("Conditional abort op can only test a scalar tensor");
    }
  }
}

static OpDefinition AbortOpDef;

static OpCreator<AbortOp> AbortOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Abort, AbortOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<AbortOp>(new AbortOp(info.opid, info.settings));
    },
    true);

} // namespace popart
