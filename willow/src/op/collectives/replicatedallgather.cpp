// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReplicatedAllGatherOp::ReplicatedAllGatherOp(const OperatorIdentifier &_opid,
                                             CommGroup group,
                                             const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, group, settings_) {}

ReplicatedAllGatherOp::ReplicatedAllGatherOp(const OperatorIdentifier &_opid,
                                             CommGroup group,
                                             const Op::Settings &settings_,
                                             TensorInfo gatheredOutInfo_)
    : CollectivesBaseOp(_opid, group, settings_),
      gatheredOutInfo(gatheredOutInfo_) {}

std::unique_ptr<Op> ReplicatedAllGatherOp::clone() const {
  return std::make_unique<ReplicatedAllGatherOp>(*this);
}

void ReplicatedAllGatherOp::setup() {
  auto globalReplicationFactor =
      getIr().getSessionOptions().getGlobalReplicationFactor();
  auto replicationFactor = globalReplicationFactor;

  if (getGCLCommGroup().replicaGroupSize > 0 &&
      (getGCLCommGroup().type == CommGroupType::Consecutive ||
       getGCLCommGroup().type == CommGroupType::Orthogonal)) {
    replicationFactor = getGCLCommGroup().replicaGroupSize;
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

} // namespace popart
