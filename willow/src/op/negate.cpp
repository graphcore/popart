// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/negate.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operators.hpp"

namespace popart {
struct OperatorIdentifier;

NegateOp::NegateOp(const OperatorIdentifier &_opid,
                   const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> NegateOp::clone() const {
  return std::make_unique<NegateOp>(*this);
}

std::vector<std::unique_ptr<Op>> NegateOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<NegateGradOp>(*this));
  return upops;
}

NegateGradOp::NegateGradOp(const NegateOp &fwdOp)
    : NegateOp(Onnx::GradOperators::NegGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> NegateGradOp::clone() const {
  return std::make_unique<NegateGradOp>(*this);
}

const std::vector<GradInOutMapper> &NegateGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), NegateOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &NegateGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), NegateOp::getInIndex()}};

  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition negateOpDef({OpDefinition::Inputs({{"X", T}}),
                                 OpDefinition::Outputs({{"Y", T}}),
                                 OpDefinition::Attributes({})});

static OpCreator<NegateOp>
    negateOpCreator(OpDefinitions({{Onnx::Operators::Neg_6, negateOpDef}}));

} // namespace

} // namespace popart
