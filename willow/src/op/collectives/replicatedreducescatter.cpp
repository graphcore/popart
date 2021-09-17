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
    CommGroup group,
    const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, group, settings_), op(op_) {}

ReplicatedReduceScatterOp::ReplicatedReduceScatterOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, CommGroup{}, settings_),
      op(CollectiveOperator::Add) {}

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

  outInfo(getOutIndex()) =
      TensorInfo(inInfo_.dataType(), {outElms}, inInfo_.shape());

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
      CollectiveOperator op = static_cast<CollectiveOperator>(
          info.attributes.getAttribute<Attributes::Int>(
              sCollectiveOperator, static_cast<int>(CollectiveOperator::Add)));
      CommGroup group = extractCommGroupFromAttrs(info.attributes);
      return std::unique_ptr<ReplicatedReduceScatterOp>(
          new ReplicatedReduceScatterOp(info.opid, op, group, info.settings));
    },
    true);

} // namespace popart
