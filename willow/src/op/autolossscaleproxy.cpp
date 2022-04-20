// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/autolossscaleproxy.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"

namespace popart {
struct OperatorIdentifier;

AutoLossScaleProxyOp::AutoLossScaleProxyOp(const OperatorIdentifier &_opid,
                                           const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> AutoLossScaleProxyOp::clone() const {
  return std::make_unique<AutoLossScaleProxyOp>(*this);
}

std::vector<std::unique_ptr<Op>> AutoLossScaleProxyOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<AutoLossScaleProxyGradOp>(*this));
  return upops;
}

AutoLossScaleProxyGradOp::AutoLossScaleProxyGradOp(
    const AutoLossScaleProxyOp &fwdOp)
    : AutoLossScaleProxyOp(Onnx::GradOperators::AutoLossScaleProxyGrad,
                           fwdOp.getSettings()) {}

std::unique_ptr<Op> AutoLossScaleProxyGradOp::clone() const {
  return std::make_unique<AutoLossScaleProxyGradOp>(*this);
}

const std::vector<GradInOutMapper> &
AutoLossScaleProxyGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(),
       AutoLossScaleProxyOp::getOutIndex(),
       GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &AutoLossScaleProxyGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AutoLossScaleProxyOp::getInIndex()}};

  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    autoLossScaleProxyOpDef({OpDefinition::Inputs({{"input", T}}),
                             OpDefinition::Outputs({{"output", T}}),
                             OpDefinition::Attributes({})});

static OpCreator<AutoLossScaleProxyOp> AutoLossScaleProxyOpCreator(
    OpDefinitions({{Onnx::CustomOperators::AutoLossScaleProxy,
                    autoLossScaleProxyOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(
          new AutoLossScaleProxyOp(info.opid, info.settings));
    },
    true);

} // namespace

} // namespace popart
