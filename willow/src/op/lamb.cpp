// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/lamb.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include "popart/commgroup.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

LambSquareOp::LambSquareOp(const Op::Settings &settings_)
    : Op(Onnx::CustomOperators::LambSquare, settings_) {}

std::unique_ptr<Op> LambSquareOp::clone() const {
  return std::make_unique<LambSquareOp>(*this);
}

void LambSquareOp::setup() { outInfo(getOutIndex()) = {DataType::FLOAT, {}}; }

ReplicatedTensorShardingIndices
LambSquareOp::getReplicatedTensorShardingIndices() const {
  return {{{LambSquareOp::getInIndex()}, {}}};
}

void LambSquareOp::configureForReplicatedTensorSharding(
    ReplicatedTensorShardingIndices indices,
    CommGroup shardingDomain) {
  configureForReplicatedTensorSharding(
      std::move(indices),
      shardingDomain.toReplicaGrouping(
          getIr().getSessionOptions().getGlobalReplicationFactor()));
}

void LambSquareOp::configureForReplicatedTensorSharding(
    ReplicatedTensorShardingIndices indices,
    const ReplicaGrouping &grouping) {
  if (indices == getReplicatedTensorShardingIndices()) {
    Tensor *out = output->tensor(LambSquareOp::getOutIndex());

    // Make sure reduction is only added once
    auto lambSqConsumers = out->consumers.getOps();
    if (!std::any_of(
            lambSqConsumers.begin(), lambSqConsumers.end(), [](Op *op) {
              return dynamic_cast<ReplicatedAllReduceOp *>(op) ||
                     dynamic_cast<ReplicatedAllReduceInplaceOp *>(op);
            })) {

      TensorId lambIntoReduceId =
          getGraph().getIr().createIntermediateTensorId(out->id);

      disconnectOutTensor(out);
      createAndConnectOutTensor(ReplicatedAllReduceOp::getOutIndex(),
                                lambIntoReduceId);
      setup();

      auto reduceOpUp = std::make_unique<ReplicatedAllReduceInplaceOp>(
          Onnx::CustomOperators::ReplicatedAllReduceInplace,
          CollectiveOperator::Add,
          grouping,
          settings);
      auto reduceOp = reduceOpUp.get();
      getGraph().moveIntoGraph(std::move(reduceOpUp));

      reduceOp->connectInTensor(ReplicatedAllReduceInplaceOp::getInIndex(),
                                lambIntoReduceId);
      reduceOp->connectOutTensor(ReplicatedAllReduceInplaceOp::getOutIndex(),
                                 out->id);

      reduceOp->setup();
    }
  } else {
    throw error("LambSquareOp::configureForReplicatedTensorSharding "
                "Unexpected input indices.");
  }
}

} // namespace popart
