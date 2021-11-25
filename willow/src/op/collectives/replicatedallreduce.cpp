// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/alias/aliasmodel.hpp>
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
  CollectivesBaseOp::appendOutlineAttributes(os);
  os.appendAttribute(sCollectiveOperator, static_cast<int>(op));
}

ReplicatedTensorShardingIndices
ReplicatedAllReduceOp::getReplicatedTensorShardingIndices() const {
  return {{{ReplicatedAllReduceOp::getInIndex()},
           {ReplicatedAllReduceOp::getOutIndex()}}};
}

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
ReplicatedAllReduceOp::fwdPropagateIsReplicaEqual(
    const AliasModel &aliasModel,
    const ReplEqInputMap &inputMap,
    ReplicaEqualAnalysisProxy &proxy) const {

  // TODO(T51589): Amend logic to be more fine-grained, taking into account
  // CommGroup settings. We should work out replica-equalness over subsets
  // of replicas instead instead of having only tracking if a tensor is
  // replica-equal for all replicas or not.

  const auto groupType = getGCLCommGroup().type;
  const auto groupSize = getGCLCommGroup().replicaGroupSize;
  const auto maxGroupSize =
      getIr().getSessionOptions().getGlobalReplicationFactor();
  const auto isLocal = (op == CollectiveOperator::Local);
  const auto isReductionOverAllReplicas =
      (groupType == CommGroupType::All) ||
      (groupType == CommGroupType::Consecutive && groupSize == maxGroupSize) ||
      (groupType == CommGroupType::Orthogonal && groupSize == maxGroupSize);

  // For all reduction methods except Local, the output should be identical
  // across replicas within a group. So outputs are equal across all replicas
  // only if the grouping includes all replicas.
  if (!isLocal && isReductionOverAllReplicas) {
    ReplEqOutputMap result;
    result[getOutIndex()] = true;
    return {result, proxy.getModifiedInputMapFromAliases(this, result)};
  } else {
    return Op::fwdPropagateIsReplicaEqual(aliasModel, inputMap, proxy);
  }
}

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

static OpDefinition ReplicatedAllReduceOpDef(
    {OpDefinition::Inputs({{"X", T}}),
     OpDefinition::Outputs({{"Y", T}}),
     OpDefinition::Attributes({{sCollectiveOperator, {"*"}},
                               {sCollectiveCommGroup, {"*"}}})});

static OpCreator<ReplicatedAllReduceOp> ReplicatedAllReduceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ReplicatedAllReduce,
                    ReplicatedAllReduceOpDef}}),
    [](const OpCreatorInfo &info) {
      CommGroup group       = extractCommGroupFromAttrs(info.attributes);
      CollectiveOperator op = static_cast<CollectiveOperator>(
          info.attributes.getAttribute<Attributes::Int>(
              sCollectiveOperator, static_cast<int>(CollectiveOperator::Add)));

      return std::unique_ptr<ReplicatedAllReduceOp>(
          new ReplicatedAllReduceOp(info.opid, op, group, info.settings));
    },
    true);

static OpDefinition ReplicatedAllReduceInplaceOpDef(
    {OpDefinition::Inputs({{"X", T}}),
     OpDefinition::Outputs({{"Y", T}}),
     OpDefinition::Attributes({{sCollectiveOperator, {"*"}},
                               {sCollectiveCommGroup, {"*"}}})});

static OpCreator<ReplicatedAllReduceInplaceOp>
    ReplicatedAllReduceInplaceOpCreator(
        OpDefinitions({{Onnx::CustomOperators::ReplicatedAllReduceInplace,
                        ReplicatedAllReduceInplaceOpDef}}),
        [](const OpCreatorInfo &info) {
          CommGroup group       = extractCommGroupFromAttrs(info.attributes);
          CollectiveOperator op = static_cast<CollectiveOperator>(
              info.attributes.getAttribute<Attributes::Int>(
                  sCollectiveOperator,
                  static_cast<int>(CollectiveOperator::Add)));

          return std::unique_ptr<ReplicatedAllReduceInplaceOp>(
              new ReplicatedAllReduceInplaceOp(
                  info.opid, op, group, info.settings));
        },
        true);

} // namespace popart
