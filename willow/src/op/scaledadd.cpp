// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/op/scaledadd.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/region.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

poprithms::memory::inplace::Proposal
ScaledAddOp::mapInplaceProposal(const AliasModel &aliasModel,
                                OperatorIdentifier opId) const {
  const std::string inplaceName = opId.type;
  auto index                    = (inplaceName.find("Rhs") != std::string::npos)
                   ? getArg1InIndex()
                   : getArg0InIndex();
  return {aliasModel.getGate(id), index};
}

std::vector<std::tuple<OperatorIdentifier, float>>
ScaledAddOp::inplacePriorityDefault() const {

  if (hasInput(ScaledAddOp::getScale0InIndex()) ==
      hasInput(ScaledAddOp::getScale1InIndex())) {
    return {{Onnx::CustomOperators::ScaledAddLhsInplace, 10},
            {Onnx::CustomOperators::ScaledAddRhsInplace, 10}};
  }

  if (hasInput(ScaledAddOp::getScale1InIndex()) && getScale0() == 1.0f) {
    return {{Onnx::CustomOperators::ScaledAddLhsInplace, 10}};
  }

  if (hasInput(ScaledAddOp::getScale0InIndex()) && getScale1() == 1.0f) {
    return {{Onnx::CustomOperators::ScaledAddRhsInplace, 10}};
  }

  return {};
}

void ScaledAddOp::growAliasModel(AliasModel &m) const {
  m.insertBinaryModifier(*this);
}

ScaledAddLhsInplaceOp::ScaledAddLhsInplaceOp(float scale_0_,
                                             float scale_1_,
                                             const Op::Settings &settings_)
    : ScaledAddOp(Onnx::CustomOperators::ScaledAddLhsInplace,
                  scale_0_,
                  scale_1_,
                  settings_) {}

ScaledAddLhsInplaceOp::ScaledAddLhsInplaceOp(const ScaledAddOp &scale_op)
    : ScaledAddOp(Onnx::CustomOperators::ScaledAddLhsInplace,
                  scale_op.getScale0(),
                  scale_op.getScale1(),
                  scale_op.getSettings()) {}

ScaledAddRhsInplaceOp::ScaledAddRhsInplaceOp(const ScaledAddOp &scale_op)
    : ScaledAddOp(Onnx::CustomOperators::ScaledAddRhsInplace,
                  scale_op.getScale0(),
                  scale_op.getScale1(),
                  scale_op.getSettings()) {}

std::unique_ptr<Op>
ScaledAddOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ScaledAddLhsInplace) {
    return std::make_unique<ScaledAddLhsInplaceOp>(*this);
  }
  if (operator_id == Onnx::CustomOperators::ScaledAddRhsInplace) {
    return std::make_unique<ScaledAddRhsInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

ScaledAddOp::ScaledAddOp(const OperatorIdentifier &_opid,
                         float scale_0_,
                         float scale_1_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), scale_0(scale_0_), scale_1(scale_1_) {}

std::unique_ptr<Op> ScaledAddOp::clone() const {
  return std::make_unique<ScaledAddOp>(*this);
}

void ScaledAddOp::setup() { outInfo(getOutIndex()) = inInfo(getArg0InIndex()); }

ReplicatedTensorShardingIndices
ScaledAddOp::getReplicatedTensorShardingIndices() const {
  return {{{ScaledAddOp::getArg0InIndex(), ScaledAddOp::getArg1InIndex()},
           {ScaledAddOp::getOutIndex()}}};
}

void ScaledAddOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("scale_0", scale_0);
  os.appendAttribute("scale_1", scale_1);
}

std::unique_ptr<Op> ScaledAddLhsInplaceOp::clone() const {
  return std::make_unique<ScaledAddLhsInplaceOp>(*this);
}

std::unique_ptr<Op> ScaledAddRhsInplaceOp::clone() const {
  return std::make_unique<ScaledAddRhsInplaceOp>(*this);
}

view::Regions ScaledAddLhsInplaceOp::modifies(InIndex index) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    return {};
  }
}

view::Regions ScaledAddRhsInplaceOp::modifies(InIndex index) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else {
    return {};
  }
}

view::Regions ScaledAddLhsInplaceOp::aliases(InIndex index, OutIndex) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    return view::Regions();
  }
}

view::Regions ScaledAddRhsInplaceOp::aliases(InIndex index, OutIndex) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else {
    return view::Regions();
  }
}

view::RegMap ScaledAddLhsInplaceOp::fwdRegMap(InIndex i, OutIndex) const {
  if (i == ScaledAddOp::getArg0InIndex()) {
    return Op::fwdRegMap(i, 0);
  }
  return [](const view::Region &r) { return view::Regions(); };
}

view::RegMap ScaledAddLhsInplaceOp::bwdRegMap(InIndex i, OutIndex) const {
  if (i == ScaledAddOp::getArg0InIndex()) {
    return Op::bwdRegMap(i, 0);
  }
  return [](const view::Region &r) { return view::Regions(); };
}

view::RegMap ScaledAddRhsInplaceOp::fwdRegMap(InIndex i, OutIndex) const {
  if (i == ScaledAddOp::getArg1InIndex()) {
    return Op::fwdRegMap(i, 0);
  }
  return [](const view::Region &r) { return view::Regions(); };
}

view::RegMap ScaledAddRhsInplaceOp::bwdRegMap(InIndex i, OutIndex) const {
  if (i == ScaledAddOp::getArg1InIndex()) {
    return Op::bwdRegMap(i, 0);
  }
  return [](const view::Region &r) { return view::Regions(); };
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT, DataType::FLOAT16};

static OpDefinition scaledAddOpDef0(
    {OpDefinition::Inputs({{"X", T}, {"Y", T}}),
     OpDefinition::Outputs({{"Z", T}}),
     OpDefinition::Attributes({{"scale0", {"*"}}, {"scale1", {"*"}}})});

static OpDefinition scaledAddOpDef1(
    {OpDefinition::Inputs({{"X", T}, {"Y", T}, {"A", T}, {"B", T}}),
     OpDefinition::Outputs({{"Z", T}}),
     OpDefinition::Attributes({{"scale0", {"*"}}, {"scale1", {"*"}}})});

static OpDefinition scaledAddOpDef2(
    {OpDefinition::Inputs({{"X", T}, {"Y", T}, {"B", T}}),
     OpDefinition::Outputs({{"Z", T}}),
     OpDefinition::Attributes({{"scale0", {"*"}}, {"scale1", {"*"}}})});

static OpCreator<ScaledAddOp> scaledAddOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::ScaledAdd, scaledAddOpDef0},
        {Onnx::CustomOperators::ScaledAdd, scaledAddOpDef1},
        {Onnx::CustomOperators::ScaledAdd, scaledAddOpDef2},
    }),
    [](const OpCreatorInfo &info) {
      float scale0 =
          info.attributes.getAttribute<Attributes::Float>("scale0", 1.0f);
      float scale1 =
          info.attributes.getAttribute<Attributes::Float>("scale1", 1.0f);
      return std::unique_ptr<Op>(
          new ScaledAddOp(info.opid, scale0, scale1, info.settings));
    },
    true);

} // namespace
} // namespace popart
