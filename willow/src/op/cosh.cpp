// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <vector>
#include <popart/op/cosh.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

CoshOp::CoshOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> CoshOp::clone() const {
  return std::make_unique<CoshOp>(*this);
}

std::vector<std::unique_ptr<Op>> CoshOp::getGradOps() {
  throw error("CoshOp should be removed by pattern 'CoshOp' before call to "
              "CoshOp::getGradOps");
}

void CoshOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition coshOpDef({OpDefinition::Inputs({
                                   {"input", T},
                               }),
                               OpDefinition::Outputs({{"output", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<CoshOp>
    coshOpCreator(OpDefinitions({{Onnx::Operators::Cosh_9, coshOpDef}}));

} // namespace

} // namespace popart
