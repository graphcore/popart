// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/erf.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
ErfOp::ErfOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> ErfOp::clone() const {
  return std::make_unique<ErfOp>(*this);
}

std::vector<std::unique_ptr<Op>> ErfOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<ErfGradOp>(*this));
  return upops;
}

ErfGradOp::ErfGradOp(const ErfOp &fwdOp)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::ErfGrad, fwdOp) {}

std::unique_ptr<Op> ErfGradOp::clone() const {
  return std::make_unique<ErfGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition erfOpDef({OpDefinition::Inputs({{"input", T}}),
                              OpDefinition::Outputs({{"output", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<ErfOp> erfOpCreator(OpDefinitions({
    {Onnx::Operators::Erf_9, erfOpDef},
}));
} // namespace

} // namespace popart
