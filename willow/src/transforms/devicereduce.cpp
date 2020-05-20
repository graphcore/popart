// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <popart/transforms/devicereduce.hpp>

namespace popart {

std::size_t DeviceReduce::id() { return typeid(DeviceReduce).hash_code(); }

TensorId DeviceReduce::generateReducedTensorId(Tensor *tensor) const {
  TensorId reducedTensor = tensor->id + "_reduced";
  return reducedTensor;
}

bool DeviceReduce::apply(Graph &graph) const {
  logging::transform::debug("Applying DeviceReduce transformation");

  for (Op *op : graph.getOpSchedule({})) {

    const bool isVar0UpdateOp = op->isConvertibleTo<SGD0VarUpdateOp>();
    if (!isVar0UpdateOp ||
        !op->input->hasIndex(SGD0VarUpdateOp::getUpdaterInIndex())) {
      continue;
    }

    Tensor *grad = op->input->tensor(SGD0VarUpdateOp::getUpdaterInIndex());

    auto settings = op->settings;

    auto replicatedAllReduceOp = std::make_unique<ReplicatedAllReduceOp>(
        Onnx::CustomOperators::ReplicatedAllReduce, settings);
    auto replicatedAllReduce      = replicatedAllReduceOp.get();
    replicatedAllReduce->fromLoss = op->fromLoss;
    replicatedAllReduce->toLoss   = op->toLoss;
    graph.moveIntoGraph(std::move(replicatedAllReduceOp));

    replicatedAllReduce->connectInTensor(ReplicatedAllReduceOp::getInIndex(),
                                         grad->id);

    TensorId reducedTensorId = generateReducedTensorId(grad);
    replicatedAllReduce->createAndConnectOutTensor(
        ReplicatedAllReduceOp::getOutIndex(), reducedTensorId);

    replicatedAllReduce->setup();

    op->disconnectInTensor(grad);
    op->connectInTensor(SGD0VarUpdateOp::getUpdaterInIndex(), reducedTensorId);

    logging::transform::trace("[DeviceReduce] {} -> {}",
                              replicatedAllReduce->debugName(),
                              op->debugName());

    // Tie the reduction operation to the SGD0VarUpdate to get the same schedule
    // behaviour as if the reduction was still integrated into SGD0VarUpdate
    graph.topoCons->insert(replicatedAllReduce, op, true);

    replicatedAllReduce->inheritPlacementAttributes(false);
  }
  return true;
}

namespace {
bool init = Transform::registerTransform(new DeviceReduce);
}

} // namespace popart
