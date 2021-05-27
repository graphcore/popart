// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <aliasmodel.hpp>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

ReplicatedAllReduceOp::ReplicatedAllReduceOp(const OperatorIdentifier &_opid,
                                             CollectiveOperator op_,
                                             CommGroup group,
                                             const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, group, settings_), op(op_) {}

ReplicatedAllReduceOp::ReplicatedAllReduceOp(const OperatorIdentifier &_opid,
                                             const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, CommGroup{}, settings_),
      op(CollectiveOperator::Add) {}

std::unique_ptr<Op> ReplicatedAllReduceOp::clone() const {
  return std::make_unique<ReplicatedAllReduceOp>(*this);
}

std::unique_ptr<Op> ReplicatedAllReduceOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ReplicatedAllReduceInplace) {
    return std::make_unique<ReplicatedAllReduceInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

void ReplicatedAllReduceOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

void ReplicatedAllReduceOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("op", static_cast<int>(op));
}

CommGroup ReplicatedAllReduceOp::getGCLCommGroup() const { return group; }

ReplicatedAllReduceInplaceOp::ReplicatedAllReduceInplaceOp(
    const OperatorIdentifier &_opid,
    CollectiveOperator op_,
    CommGroup group,
    const Op::Settings &settings_)
    : ReplicatedAllReduceOp(_opid, op_, group, settings_) {}

ReplicatedAllReduceInplaceOp::ReplicatedAllReduceInplaceOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_)
    : ReplicatedAllReduceOp(_opid, settings_) {}

ReplicatedAllReduceInplaceOp::ReplicatedAllReduceInplaceOp(
    const ReplicatedAllReduceOp &rop)
    : ReplicatedAllReduceInplaceOp(
          Onnx::CustomOperators::ReplicatedAllReduceInplace,
          rop.getCollectiveOp(),
          rop.getGCLCommGroup(),
          rop.getSettings()) {}

view::Regions ReplicatedAllReduceInplaceOp::modifies(InIndex index) const {
  if (index == getInIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else {
    throw error("Invalid index passed to modifies");
  }
}

view::Regions ReplicatedAllReduceInplaceOp::aliases(InIndex in,
                                                    OutIndex out) const {

  if (in != out) {
    throw error("In index and out index not equal");
  }
  if (in == getInIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    throw error("Invalid index passed to aliases");
  }
}

void ReplicatedAllReduceOp::growAliasModel(AliasModel &m) const {
  m.insertUnaryModifier0(*this);
}

std::unique_ptr<Op> ReplicatedAllReduceInplaceOp::clone() const {
  return std::make_unique<ReplicatedAllReduceInplaceOp>(*this);
}

void ReplicatedAllReduceInplaceOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    ReplicatedAllReduceOpDef({OpDefinition::Inputs({{"X", T}}),
                              OpDefinition::Outputs({{"Y", T}}),
                              OpDefinition::Attributes({{"op", {"*"}}})});

static OpCreator<ReplicatedAllReduceOp> ReplicatedAllReduceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ReplicatedAllReduce,
                    ReplicatedAllReduceOpDef}}),
    [](const OpCreatorInfo &info) {
      CommGroup group       = extractCommGroupFromAttrs(info.attributes);
      CollectiveOperator op = static_cast<CollectiveOperator>(
          info.attributes.getAttribute<Attributes::Int>(
              "op", static_cast<int>(CollectiveOperator::Add)));

      return std::unique_ptr<ReplicatedAllReduceOp>(
          new ReplicatedAllReduceOp(info.opid, op, group, info.settings));
    },
    true);

static OpDefinition ReplicatedAllReduceInplaceOpDef(
    {OpDefinition::Inputs({{"X", T}}),
     OpDefinition::Outputs({{"Y", T}}),
     OpDefinition::Attributes({{"op", {"*"}}})});

static OpCreator<ReplicatedAllReduceInplaceOp>
    ReplicatedAllReduceInplaceOpCreator(
        OpDefinitions({{Onnx::CustomOperators::ReplicatedAllReduceInplace,
                        ReplicatedAllReduceInplaceOpDef}}),
        [](const OpCreatorInfo &info) {
          CommGroup group       = extractCommGroupFromAttrs(info.attributes);
          CollectiveOperator op = static_cast<CollectiveOperator>(
              info.attributes.getAttribute<Attributes::Int>(
                  "op", static_cast<int>(CollectiveOperator::Add)));

          return std::unique_ptr<ReplicatedAllReduceInplaceOp>(
              new ReplicatedAllReduceInplaceOp(
                  info.opid, op, group, info.settings));
        },
        true);

} // namespace popart
