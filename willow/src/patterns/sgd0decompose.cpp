// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnxutil.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/sgd0combo.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/slice.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/sgd0decompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool SGD0Decompose::matches(Op *op) const {
  return op->isConvertibleTo<SGD0ComboOp>();
}

std::vector<const Tensor *> SGD0Decompose::touches(Op *) const { return {}; }

bool SGD0Decompose::apply(Op *op) const {

  auto &graph = op->getGraph();

  // matches must have verified the correctness before this call
  auto combo = static_cast<SGD0ComboOp *>(op);

  Tensor *weightGrad =
      combo->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  Tensor *weight    = combo->inTensor(VarUpdateOp::getVarToUpdateInIndex());
  Tensor *newWeight = combo->outTensor(VarUpdateOp::getUpdatedVarOutIndex());

  TensorId weightGradId    = weightGrad->id;
  TensorId weightId        = weight->id;
  TensorId updatedWeightId = newWeight->id;

  auto weightInfo  = weight->info;
  auto weightShape = weightInfo.shape();

  // Accumulator
  TensorId accumId = reservedAccumPrefix() + weightId;
  if (combo->withGradAccum) {
    addStateTensor(
        graph, accumId, weightShape, combo->accumType, VariableSettings());
  }

  TensorId gradIntoAccumId  = weightGradId;
  TensorId gradIntoUpdateId = weightGradId;
  TensorId finalGradId      = reservedFinalReducedGradPrefix() + weightId;

  if (combo->reductionType == OptimizerReductionType::GradReduce) {
    TensorId reducedId = gradReduce(
        graph, combo, weightGradId, !combo->withGradAccum ? finalGradId : "");
    gradIntoAccumId  = reducedId;
    gradIntoUpdateId = reducedId;
  }

  Op *zeroAccum = nullptr;
  if (combo->withGradAccum) {
    gradIntoUpdateId =
        gradAccum(graph,
                  combo,
                  accumId,
                  gradIntoAccumId,
                  combo->reductionType == OptimizerReductionType::AccumReduce,
                  finalGradId);
    if (!runningMeanReduction(graph)) {
      zeroAccum = zeroAccumulator(graph, combo, {}, accumId);
    }
  }

  Op *varUpdate = varUpdateAndEraseCombo(
      graph, combo, weightId, gradIntoUpdateId, updatedWeightId);

  // Zero the gradient accumulator after updating the 1st momentum term
  // ready for next step
  if (zeroAccum != nullptr) {
    graph.topoCons->insert(varUpdate, zeroAccum);
  }

  return true;
}

Op *SGD0Decompose::varUpdateAndEraseCombo(
    Graph &graph,
    SGD0ComboOp *combo,
    const TensorId &weightId,
    const TensorId &gradIntoUpdateId,
    const TensorId &updatedWeightId) const {

  auto sgd0VarUpdate = graph.createOp<SGD0VarUpdateOp>(
      combo->initSlr0,
      combo->initWdsf0,
      Op::Settings(
          graph, combo->name() + "_var_update", combo->settings.debugInfoId));
  transferBaseProperties(combo, sgd0VarUpdate);

  sgd0VarUpdate->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                                 weightId);

  sgd0VarUpdate->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                                 gradIntoUpdateId);

  if (!combo->initSlr0.isConst()) {
    sgd0VarUpdate->connectInTensor(SGD0VarUpdateOp::getSlr0InIndex(),
                                   combo->inId(SGD0ComboOp::getSlr0InIndex()));
  }

  if (!combo->initWdsf0.isConst()) {
    sgd0VarUpdate->connectInTensor(SGD0VarUpdateOp::getWdsf0InIndex(),
                                   combo->inId(SGD0ComboOp::getWdsf0InIndex()));
  }

  if (combo->withGradAccum) {
    sgd0VarUpdate->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    sgd0VarUpdate->setExecutionPhase({});
    sgd0VarUpdate->settings.schedulePriority = 0.0;
  } else {
    graph.topoCons->transfer(combo, sgd0VarUpdate);
  }

  // Deleting combo op now, so that its output can be re-connected.
  combo->disconnectAllInputs();
  combo->disconnectAllOutputs();
  graph.eraseOp(combo->id);

  // (4)
  sgd0VarUpdate->connectOutTensor(SGD0VarUpdateOp::getUpdatedVarOutIndex(),
                                  updatedWeightId);
  sgd0VarUpdate->setup();

  return sgd0VarUpdate;
}

namespace {
// Not registering this pattern, as we want it to run at a special time
static AddPatternName<SGD0Decompose> registerName("SGD0Decompose");
} // namespace

} // namespace popart
