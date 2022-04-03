// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

ReplicatedReduceScatterOp::ReplicatedReduceScatterOp(
    const OperatorIdentifier &_opid,
    CollectiveOperator op_,
    CommGroup group_,
    bool configureOutputForReplicatedTensorSharding_,
    const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, group_, settings_), op(op_),
      configureOutputForReplicatedTensorSharding(
          configureOutputForReplicatedTensorSharding_) {}

ReplicatedReduceScatterOp::ReplicatedReduceScatterOp(
    const OperatorIdentifier &_opid,
    CollectiveOperator op_,
    CommGroup group_,
    const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, group_, settings_), op(op_),
      configureOutputForReplicatedTensorSharding(false) {}

ReplicatedReduceScatterOp::ReplicatedReduceScatterOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, CommGroup{}, settings_),
      op(CollectiveOperator::Add),
      configureOutputForReplicatedTensorSharding(false) {}

std::unique_ptr<Op> ReplicatedReduceScatterOp::clone() const {
  return std::make_unique<ReplicatedReduceScatterOp>(*this);
}

void ReplicatedReduceScatterOp::setup() {

  const auto &inInfo_ = inInfo(getInIndex());

  auto globalReplicationFactor =
      getIr().getSessionOptions().getGlobalReplicationFactor();
  auto replicationFactor = globalReplicationFactor;
  int64_t nelms          = inInfo_.nelms();

  if (getGCLCommGroup().replicaGroupSize > 0 &&
      (getGCLCommGroup().type == CommGroupType::Consecutive ||
       getGCLCommGroup().type == CommGroupType::Orthogonal)) {
    replicationFactor = getGCLCommGroup().replicaGroupSize;
  }

  // ceil(numElements / replicationFactor)
  auto outElms = (nelms + replicationFactor - 1) / replicationFactor;

  Shape metaShape;
  if (isConfigureOutputForReplicatedTensorSharding()) {
    metaShape = inInfo_.shape();
  }

  outInfo(getOutIndex()) = TensorInfo(inInfo_.dataType(), {outElms}, metaShape);

  logging::op::trace("[ReplicatedReduceScatterOp] Global replication factor: "
                     "{}, sharding factor: {}",
                     globalReplicationFactor,
                     replicationFactor);
}

void ReplicatedReduceScatterOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  CollectivesBaseOp::appendOutlineAttributes(os);
  os.appendAttribute(sCollectiveOperator, static_cast<int>(op));
}

ReplicatedTensorShardingIndices
ReplicatedReduceScatterOp::getReplicatedTensorShardingIndices() const {
  return {{{}, {ReplicatedReduceScatterOp::getOutIndex()}}};
}

bool ReplicatedReduceScatterOp::isConfigureOutputForReplicatedTensorSharding()
    const {
  return configureOutputForReplicatedTensorSharding ||
         hasInput(ReplicatedReduceScatterOp::getCollectiveLinkedIndex()) ||
         !outInfo(ReplicatedReduceScatterOp::getOutIndex()).metaShape().empty();
}

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
ReplicatedReduceScatterOp::fwdPropagateIsReplicaEqual(
    const AliasModel &aliasModel,
    const ReplEqInputMap &inputMap,
    ReplicaEqualAnalysisProxy &proxy) const {

  // TODO(T51589): Amend logic to be more fine-grained, taking into account
  // CommGroup settings. We should work out replica-equalness over subsets
  // of replicas instead instead of having only tracking if a tensor is
  // replica-equal for all replicas or not.

  const auto groupType = getGCLCommGroup().type;
  const auto groupSize = getGCLCommGroup().replicaGroupSize;
  const auto isLocal   = (op == CollectiveOperator::Local);
  const auto isReductionOverOneReplica =
      (groupType == CommGroupType::None) ||
      (groupType == CommGroupType::Consecutive && groupSize == 1) ||
      (groupType == CommGroupType::Orthogonal && groupSize == 1);

  // If a local reduction or a scatter over multiple replicas, the output is
  // definitely non-equal.
  if (isLocal || !isReductionOverOneReplica) {
    ReplEqOutputMap result;
    result[getOutIndex()] = false;
    return {result, proxy.getModifiedInputMapFromAliases(this, result)};
  } else {
    return Op::fwdPropagateIsReplicaEqual(aliasModel, inputMap, proxy);
  }
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition ReplicatedReduceScatterOpDef(
    {OpDefinition::Inputs({{"X", T}}),
     OpDefinition::Outputs({{"Y", T}}),
     OpDefinition::Attributes({{sCollectiveOperator, {"*"}},
                               {sCollectiveCommGroup, {"*"}}})});

static OpCreator<ReplicatedReduceScatterOp> ReplicatedReduceScatterOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ReplicatedReduceScatter,
                    ReplicatedReduceScatterOpDef}}),
    [](const OpCreatorInfo &info) {
      CommGroup group       = extractCommGroupFromAttrs(info.attributes);
      CollectiveOperator op = static_cast<CollectiveOperator>(
          info.attributes.getAttribute<Attributes::Int>(
              sCollectiveOperator, static_cast<int>(CollectiveOperator::Add)));
      bool replicatedTensorSharding =
          static_cast<bool>(info.attributes.getAttribute<Attributes::Int>(
              sReplicatedTensorSharding, 0));
      return std::unique_ptr<ReplicatedReduceScatterOp>(
          new ReplicatedReduceScatterOp(
              info.opid, op, group, replicatedTensorSharding, info.settings));
    },
    true);

} // namespace popart
