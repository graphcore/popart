// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/flatten.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/slice.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/sgd0decompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool SGD0Decompose::matches(Op *op) const {
  return op->isConvertibleTo<SGD0VarUpdateOp>();
}

std::vector<const Tensor *> SGD0Decompose::touches(Op *) const { return {}; }

bool SGD0Decompose::apply(Op *op) const {

  auto &ir    = op->getIr();
  auto &graph = op->getGraph();

  // matches must have verified the correctness before this call
  auto sgd0 = static_cast<SGD0VarUpdateOp *>(op);

  if (sgd0->getReductionType() == OptimizerReductionType::GradReduce) {

    InIndex inIndex = SGD0VarUpdateOp::getUpdaterInIndex();
    Tensor *grad    = sgd0->input->tensor(inIndex);

    auto reduceOpUp = std::make_unique<ReplicatedAllReduceInplaceOp>(
        Onnx::CustomOperators::ReplicatedAllReduceInplace,
        Op::Settings(graph, sgd0->name() + "_reduce"));
    auto reduceOp = reduceOpUp.get();
    transferBaseProperties(sgd0, reduceOp);
    graph.moveIntoGraph(std::move(reduceOpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            grad->id,
                            reduceOp->str(),
                            ReplicatedAllReduceInplaceOp::getInIndex());
    reduceOp->connectInTensor(ReplicatedAllReduceInplaceOp::getInIndex(),
                              grad->id);

    TensorId reducedTensorId = grad->id + "_reduced";

    reduceOp->createAndConnectOutTensor(
        ReplicatedAllReduceInplaceOp::getOutIndex(), reducedTensorId);

    reduceOp->setup();

    logging::transform::trace(
        "[SGD0Decompose] {} -> {}", reduceOp->debugName(), op->debugName());

    // Tie the reduction operation to the SGD0VarUpdate to get the same schedule
    // behaviour as if the reduction was still integrated into SGD0VarUpdate
    graph.topoCons->insert(reduceOp, sgd0, true);

    reduceOp->inheritPlacementAttributes(false);
  }

  return true;
}

namespace {
// Not registering this pattern, as we want it to run at a special time
static AddPatternName<SGD0Decompose> registerName("SGD0Decompose");
} // namespace

} // namespace popart
