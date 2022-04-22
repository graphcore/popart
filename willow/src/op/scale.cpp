// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <popart/op/scale.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
ScaleOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::ScaleInplace, 10}};
}

ScaleInplaceOp::ScaleInplaceOp(const ScaleOp &scale_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::ScaleInplace,
                                scale_op.getSettings()),
      scale_factor(scale_op.getScaleFactor()) {}

ScaleInplaceOp::ScaleInplaceOp(const OperatorIdentifier &_opid,
                               float scale_,
                               const Op::Settings &settings_)
    : ElementWiseInplaceUnaryOp(_opid, settings_), scale_factor(scale_) {}

std::unique_ptr<Op>
ScaleOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ScaleInplace) {
    return std::make_unique<ScaleInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

ScaleOp::ScaleOp(const OperatorIdentifier &_opid,
                 float scale_,
                 const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), scale_factor(scale_) {}

std::unique_ptr<Op> ScaleOp::clone() const {
  return std::make_unique<ScaleOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScaleOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<ScaleGradOp>(*this));
  return upops;
}

float ScaleOp::getScaleFactor() const { return scale_factor; }
float ScaleInplaceOp::getScaleFactor() const { return scale_factor; }

void ScaleOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("scale", scale_factor);
}

void ScaleInplaceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("scale", scale_factor);
}

std::unique_ptr<Op> ScaleInplaceOp::clone() const {
  return std::make_unique<ScaleInplaceOp>(*this);
}

// A scale with a scale factor of +1 can be replaced by identity
bool ScaleOp::canBeReplacedByIdentity() const {
  return getScaleFactor() == 1.0f;
}

ScaleGradOp::ScaleGradOp(const ScaleOp &fwdOp)
    : ScaleOp(Onnx::GradOperators::ScaleGrad,
              fwdOp.getScaleFactor(),
              fwdOp.getSettings()) {}

std::unique_ptr<Op> ScaleGradOp::clone() const {
  return std::make_unique<ScaleGradOp>(*this);
}

const std::vector<GradInOutMapper> &ScaleGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ScaleOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &ScaleGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ScaleOp::getInIndex()}};

  return outInfo;
}

namespace {

// Does this support FLOAT16. defs.cc says no
static OpDefinition::DataTypes T = {DataType::FLOAT};

static OpDefinition scaleOpDef({OpDefinition::Inputs({{"X", T}}),
                                OpDefinition::Outputs({{"Y", T}}),
                                OpDefinition::Attributes({{"scale", {"*"}}})});

static OpCreator<ScaleOp> scaleOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Scale_1, scaleOpDef}}),
    [](const OpCreatorInfo &info) {
      float scale =
          info.attributes.getAttribute<Attributes::Float>("scale", 1.0f);

      return std::unique_ptr<Op>(new ScaleOp(info.opid, scale, info.settings));
    },
    true);

} // namespace
} // namespace popart
