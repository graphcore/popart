// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/reciprocal.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReciprocalOp::ReciprocalOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> ReciprocalOp::clone() const {
  return std::make_unique<ReciprocalOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReciprocalOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  upops.emplace_back(std::make_unique<ReciprocalGradOp>(*this));
  return upops;
}

ReciprocalGradOp::ReciprocalGradOp(const ReciprocalOp &op_)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::ReciprocalGrad,
                                      op_) {}

std::unique_ptr<Op> ReciprocalGradOp::clone() const {
  return std::make_unique<ReciprocalGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition receiprocalOpDef({OpDefinition::Inputs({{"X", T}}),
                                      OpDefinition::Outputs({{"Y", T}}),
                                      OpDefinition::Attributes({})});

static OpCreator<ReciprocalOp> receiprocalOpCreator(
    OpDefinitions({{Onnx::Operators::Reciprocal_6, receiprocalOpDef}}));

} // namespace

} // namespace popart
