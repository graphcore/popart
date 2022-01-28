// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/tensorremap.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

TensorRemapOp::TensorRemapOp(const OperatorIdentifier &_opid,
                             const TensorRemapType &remap_type_,
                             const Op::Settings &settings_)
    : Op(_opid, settings_), remap_type(remap_type_) {}

TensorRemapOp::TensorRemapOp(const TensorRemapOp &op_)
    : TensorRemapOp(op_.opid, op_.getTensorRemapType(), op_.settings) {}

std::unique_ptr<Op> TensorRemapOp::clone() const {
  return std::make_unique<TensorRemapOp>(*this);
}

void TensorRemapOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

std::vector<std::unique_ptr<Op>> TensorRemapOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  if (remap_type == TensorRemapType::Fwd) {
    upops.emplace_back(std::make_unique<IdentityGradOp>(settings));
  } else {
    upops.emplace_back(std::make_unique<TensorRemapOp>(*this));
  }
  return upops;
}

const std::vector<GradInOutMapper> &TensorRemapOp::gradInputInfo() const {
  if (remap_type == TensorRemapType::FwdBwdReverse) {
    static const std::vector<GradInOutMapper> inInfo = {
        {getInIndex(), TensorRemapOp::getOutIndex(), GradOpInType::GradOut},
        {getRefInIndex(), TensorRemapOp::getInIndex(), GradOpInType::In}};
    return inInfo;
  } else {
    static const std::vector<GradInOutMapper> inInfo = {
        {getInIndex(), TensorRemapOp::getOutIndex(), GradOpInType::GradOut}};
    return inInfo;
  }
}

const std::map<int, int> &TensorRemapOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), TensorRemapOp::getInIndex()}};
  return outInfo;
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition tensorRemapOpDef1({OpDefinition::Inputs({{"input", T}}),
                                       OpDefinition::Outputs({{"output", T}}),
                                       OpDefinition::Attributes({
                                           {"remap_type", {"*"}},
                                       })});

static OpDefinition
    tensorRemapOpDef2({OpDefinition::Inputs({{"input", T}, {"ref", T}}),
                       OpDefinition::Outputs({{"output", T}}),
                       OpDefinition::Attributes({
                           {"remap_type", {"*"}},
                       })});

static OpCreator<TensorRemapOp> initOpCreator(
    OpDefinitions({{Onnx::CustomOperators::TensorRemap_1, tensorRemapOpDef1},
                   {Onnx::CustomOperators::TensorRemap_1, tensorRemapOpDef2}}),
    [](const OpCreatorInfo &info) {
      TensorRemapType remap_type = static_cast<TensorRemapType>(
          info.attributes.getAttribute<Attributes::Int>("remap_type"));
      return std::unique_ptr<TensorRemapOp>(
          new TensorRemapOp(info.opid, remap_type, info.settings));
    },
    true);

} // namespace popart
