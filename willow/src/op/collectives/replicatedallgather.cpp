// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

#include "popart/analysis/replicaequal/replicaequalanalysisproxy.hpp"
#include "popart/commgroup.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class AliasModel;
struct OperatorIdentifier;

ReplicatedAllGatherOp::ReplicatedAllGatherOp(const OperatorIdentifier &_opid,
                                             CommGroup group_,
                                             const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, group_, settings_) {}

ReplicatedAllGatherOp::ReplicatedAllGatherOp(const OperatorIdentifier &_opid,
                                             CommGroup group_,
                                             const Op::Settings &settings_,
                                             TensorInfo gatheredOutInfo_)
    : CollectivesBaseOp(_opid, group_, settings_),
      gatheredOutInfo(gatheredOutInfo_) {}

std::unique_ptr<Op> ReplicatedAllGatherOp::clone() const {
  return std::make_unique<ReplicatedAllGatherOp>(*this);
}

void ReplicatedAllGatherOp::setup() {
  auto globalReplicationFactor =
      getIr().getSessionOptions().getGlobalReplicationFactor();
  auto replicationFactor = globalReplicationFactor;

  auto shardingDomain = getGCLCommGroup();
  if (shardingDomain.replicaGroupSize > 0) {
    if (shardingDomain.type == CommGroupType::Consecutive) {
      replicationFactor = shardingDomain.replicaGroupSize;
    } else if (shardingDomain.type == CommGroupType::Orthogonal) {
      replicationFactor = replicationFactor / shardingDomain.replicaGroupSize;
    }
  } else if (shardingDomain.type == CommGroupType::None) {
    replicationFactor = 1;
  }

  DataType type =
      inTensor(ReplicatedAllGatherOp::getInIndex())->info.dataType();
  Shape shape = gatheredOutInfo.shape();
  if (gatheredOutInfo.shape().empty()) {
    gatheredOutInfo = inInfo(ReplicatedAllGatherOp::getInIndex());
    Shape new_shape(1, replicationFactor * gatheredOutInfo.nelms());
    shape = new_shape;
  }
  gatheredOutInfo.set(type, shape);
  outInfo(getOutIndex()) = gatheredOutInfo;

  logging::op::trace("[ReplicatedAllGatherOp] Global replication factor: {}, "
                     "sharding factor: {}",
                     globalReplicationFactor,
                     replicationFactor);
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition ReplicatedAllGatherOpDef({OpDefinition::Inputs({{"X", T}}),
                                              OpDefinition::Outputs({{"Y", T}}),
                                              OpDefinition::Attributes({})});

static OpCreator<ReplicatedAllGatherOp> ReplicatedAllGatherOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ReplicatedAllGather,
                    ReplicatedAllGatherOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<ReplicatedAllGatherOp>(
          new ReplicatedAllGatherOp(info.opid,
                                    extractCommGroupFromAttrs(info.attributes),
                                    info.settings));
    },
    true);

ReplicatedTensorShardingIndices
ReplicatedAllGatherOp::getReplicatedTensorShardingIndices() const {
  return {{{ReplicatedAllGatherOp::getInIndex()}, {}}};
}

bool ReplicatedAllGatherOp::isConfigureOutputForReplicatedTensorSharding()
    const {
  return hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex()) ||
         !inInfo(ReplicatedAllGatherOp::getInIndex()).metaShape().empty();
}

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
ReplicatedAllGatherOp::fwdPropagateIsReplicaEqual(
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
  const auto isReductionOverAllReplicas =
      (groupType == CommGroupType::All) ||
      (groupType == CommGroupType::Consecutive && groupSize == maxGroupSize) ||
      (groupType == CommGroupType::Orthogonal && groupSize == maxGroupSize);

  // The output should be identical across replicas within a group. So outputs
  // are equal across all replicas only if the grouping includes all replicas.
  if (isReductionOverAllReplicas) {
    ReplEqOutputMap result;
    result[getOutIndex()] = true;
    return {result, proxy.getModifiedInputMapFromAliases(this, result)};
  } else {
    return Op::fwdPropagateIsReplicaEqual(aliasModel, inputMap, proxy);
  }
}

} // namespace popart
