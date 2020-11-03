// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ces/concatce.hpp>
#include <popart/ces/flattence.hpp>
#include <popart/ces/slicece.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/adamcombo.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/slice.hpp>
#include <popart/patterns/adamdecompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool AdamDecompose::matches(Op *op) const {
  return op->isConvertibleTo<AdamComboOp>();
}

std::vector<const Tensor *> AdamDecompose::touches(Op *) const { return {}; }

bool AdamDecompose::apply(Op *op) const {

  auto &ir    = op->getIr();
  auto &graph = op->getGraph();

  // matches must have verified the correctness before this call
  auto combo = static_cast<AdamComboOp *>(op);

  Tensor *weightGrad =
      combo->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  Tensor *weight    = combo->inTensor(VarUpdateOp::getVarToUpdateInIndex());
  Tensor *newWeight = combo->outTensor(VarUpdateOp::getUpdatedVarOutIndex());

  TensorId weightGradId    = weightGrad->id;
  TensorId weightId        = weight->id;
  TensorId updatedWeightId = newWeight->id;

  // Qualified tensor names for key tensors of the Adam/Lamb optimizers
  auto stepId        = reservedStepPrefix() + weightId;
  auto accumId       = reservedAccumPrefix() + weightId;
  auto accl1Id       = reservedAccl1Prefix() + weightId;
  auto accl2Id       = reservedAccl2Prefix() + weightId;
  auto adamUpdaterId = reservedAdamUpdaterPrefix() + weightId;
  auto lambR1SqId    = reservedLambR1SqPrefix() + weightId;
  auto lambR2SqId    = reservedLambR2SqPrefix() + weightId;

  if (combo->mode == AdamMode::Adam || combo->mode == AdamMode::Lamb ||
      combo->mode == AdamMode::AdaMax) {
    // Step
    addStateTensor<float>(graph, stepId, TensorInfo(DataType::FLOAT, {}));
    storeTensor(ir, stepId);
  }

  if (weightGrad->info.dataType() != weight->info.dataType()) {
    throw error("Currently, weight and weight gradient should have the same "
                "type in AdamDecompose, this is outstanding work");
  }

  auto weightInfo  = weight->info;
  auto weightShape = weightInfo.shape();

  // Accumulator
  if (combo->withGradAccum) {
    addStateTensor(graph, accumId, weightShape, combo->accumType);
  }

  // 1st momentum (accl1)
  addStateTensor(graph, accl1Id, weightShape, combo->accl1Type);

  // 2nd momentum (accl2)
  addStateTensor(graph, accl2Id, weightShape, combo->accl2Type);

  TensorId gradIntoAcclId  = weightGradId;
  TensorId gradIntoAccumId = weightGradId;

  if (combo->reductionType == OptimizerReductionType::GradReduce) {
    TensorId reducedId = gradReduce(graph, combo, weightGradId);
    gradIntoAcclId     = reducedId;
    gradIntoAccumId    = reducedId;
  }

  // Gradient accumulation
  if (combo->withGradAccum) {
    gradIntoAcclId =
        gradAccum(graph,
                  combo,
                  accumId,
                  gradIntoAccumId,
                  combo->reductionType == OptimizerReductionType::AccumReduce,
                  combo->withGradAccum);
  }

  // Cast if accumulator is fp16, and optimizer state is fp32.
  if (combo->accumType == DataType::FLOAT16 &&
      combo->accl1Type == DataType::FLOAT &&
      combo->accl2Type == DataType::FLOAT) {
    gradIntoAcclId =
        gradCast(graph, combo, gradIntoAcclId, combo->withGradAccum);
  }

  // Gradient unscaling
  TensorId gsId =
      combo->initGs.isConst() ? "" : combo->inId(AdamComboOp::getGsInIndex());
  gradIntoAcclId = gradUnscale(
      graph, combo, combo->initGs, gsId, gradIntoAcclId, combo->withGradAccum);

  // L2 regularization
  if (combo->decayMode == WeightDecayMode::L2Regularization) {
    TensorId wdId =
        combo->initWd.isConst() ? "" : combo->inId(AdamComboOp::getWdInIndex());
    gradIntoAcclId = regularizeL2(graph,
                                  combo,
                                  combo->initWd,
                                  wdId,
                                  weightId,
                                  gradIntoAcclId,
                                  combo->withGradAccum);
  }

  // 1st momentum
  auto accl1              = accl(graph,
                    combo,
                    accl1Id,
                    gradIntoAcclId,
                    AccumulationType::MovingAverage,
                    combo->initB1,
                    combo->initB1.isConst()
                        ? ""
                        : combo->inId(AdamComboOp::getBeta1InIndex()),
                    "_accl1",
                    combo->withGradAccum);
  Op *accl1Op             = accl1.first;
  TensorId updatedAccl1Id = accl1.second;

  // 2nd momentum
  auto accl2 = accl(
      graph,
      combo,
      accl2Id,
      gradIntoAcclId,
      combo->mode == AdamMode::AdaMax ? AccumulationType::Infinity
                                      : AccumulationType::MovingAverageSquare,
      combo->initB2,
      combo->initB2.isConst() ? ""
                              : combo->inId(AdamComboOp::getBeta2InIndex()),
      "_accl1",
      combo->withGradAccum);
  Op *accl2Op             = accl2.first;
  TensorId updatedAccl2Id = accl2.second;

  // The accumulator updater
  if (combo->withGradAccum) {
    accumUpdate(graph, combo, {accl1Op, accl2Op}, accumId);
  }

  // Adam updater term
  auto adamUpdOpUp = std::make_unique<AdamUpdaterOp>(
      combo->mode,
      combo->decayMode == WeightDecayMode::Decay ? combo->initWd
                                                 : OptimizerValue(0.0f, true),
      combo->initB1,
      combo->initB2,
      combo->initEps,
      Op::Settings(graph, combo->name() + "_adamupdater"));
  auto adamUpdOp = adamUpdOpUp.get();
  transferBaseProperties(combo, adamUpdOp);
  graph.moveIntoGraph(std::move(adamUpdOpUp));

  if (combo->decayMode == WeightDecayMode::Decay &&
      (!combo->initWd.isConst() || combo->initWd.val() > 0.0f)) {
    // Weight (for weight decay)
    logging::pattern::trace("Connecting input {} to {} at {}",
                            weightId,
                            adamUpdOp->str(),
                            AdamUpdaterOp::getVarInIndex());
    adamUpdOp->connectInTensor(AdamUpdaterOp::getVarInIndex(), weightId);
  }

  // 1st momentum
  logging::pattern::trace("Connecting input {} to {} at {}",
                          updatedAccl1Id,
                          adamUpdOp->str(),
                          AdamUpdaterOp::getAccl1InIndex());
  adamUpdOp->connectInTensor(AdamUpdaterOp::getAccl1InIndex(), updatedAccl1Id);

  // 2nd momentum
  logging::pattern::trace("Connecting input {} to {} at {}",
                          updatedAccl2Id,
                          adamUpdOp->str(),
                          AdamUpdaterOp::getAccl2InIndex());
  adamUpdOp->connectInTensor(AdamUpdaterOp::getAccl2InIndex(), updatedAccl2Id);

  if (combo->mode == AdamMode::Adam || combo->mode == AdamMode::Lamb ||
      combo->mode == AdamMode::AdaMax) {
    // step
    logging::pattern::trace("Connecting input {} to {} at {}",
                            stepId,
                            adamUpdOp->str(),
                            AdamUpdaterOp::getStepInIndex());
    adamUpdOp->connectInTensor(AdamUpdaterOp::getStepInIndex(), stepId);
  }

  // Optimizer parameters
  if (!combo->initWd.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getWdInIndex(),
                               combo->inId(AdamComboOp::getWdInIndex()));
  }
  if (!combo->initB1.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getBeta1InIndex(),
                               combo->inId(AdamComboOp::getBeta1InIndex()));
  }
  if (!combo->initB2.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getBeta2InIndex(),
                               combo->inId(AdamComboOp::getBeta2InIndex()));
  }
  if (!combo->initEps.isConst()) {
    adamUpdOp->connectInTensor(AdamUpdaterOp::getEpsInIndex(),
                               combo->inId(AdamComboOp::getEpsInIndex()));
  }

  // Updater term
  adamUpdOp->createAndConnectOutTensor(AdamUpdaterOp::getUpdaterOutIndex(),
                                       adamUpdaterId);
  adamUpdOp->setup();
  if (combo->withGradAccum) {
    adamUpdOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    adamUpdOp->setExecutionPhase({});
    adamUpdOp->settings.schedulePriority = 0.0;
  }

  // Lamb R1 & R2
  if (combo->mode == AdamMode::Lamb || combo->mode == AdamMode::LambNoBias) {
    auto lambR1OpUp = std::make_unique<LambSquareOp>(
        Op::Settings(graph, combo->name() + "_lamb1"));
    auto lambR1Op = lambR1OpUp.get();
    transferBaseProperties(combo, lambR1Op);
    graph.moveIntoGraph(std::move(lambR1OpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            weightId,
                            lambR1Op->str(),
                            LambSquareOp::getInIndex());
    lambR1Op->connectInTensor(LambSquareOp::getInIndex(), weightId);
    lambR1Op->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                        lambR1SqId);
    lambR1Op->setup();

    auto lambR2OpUp = std::make_unique<LambSquareOp>(
        Op::Settings(graph, combo->name() + "_lamb2"));
    auto lambR2Op = lambR2OpUp.get();
    transferBaseProperties(combo, lambR2Op);
    graph.moveIntoGraph(std::move(lambR2OpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            adamUpdaterId,
                            lambR2Op->str(),
                            LambSquareOp::getInIndex());
    lambR2Op->connectInTensor(LambSquareOp::getInIndex(), adamUpdaterId);
    lambR2Op->createAndConnectOutTensor(LambSquareOp::getOutIndex(),
                                        lambR2SqId);
    lambR2Op->setup();

    if (combo->withGradAccum) {
      lambR1Op->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
      lambR1Op->setExecutionPhase({});
      lambR1Op->settings.schedulePriority = 0.0;
      lambR2Op->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
      lambR2Op->setExecutionPhase({});
      lambR2Op->settings.schedulePriority = 0.0;
    }
  }

  // Var update
  auto adamVarUpdOpUp = std::make_unique<AdamVarUpdateOp>(
      weightId,
      combo->initLr,
      combo->initMwn,
      Op::Settings(graph, combo->name() + "_var_update"));
  auto adamVarUpdOp = adamVarUpdOpUp.get();
  transferBaseProperties(combo, adamVarUpdOp);
  graph.moveIntoGraph(std::move(adamVarUpdOpUp));

  // Weight
  logging::pattern::trace("Connecting input {} to {} at {}",
                          weightId,
                          adamVarUpdOp->str(),
                          AdamVarUpdateOp::getVarToUpdateInIndex());
  adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getVarToUpdateInIndex(),
                                weightId);

  // Updater
  logging::pattern::trace("Connecting input {} to {} at {}",
                          adamUpdaterId,
                          adamVarUpdOp->str(),
                          AdamVarUpdateOp::getUpdaterInIndex());
  adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getUpdaterInIndex(),
                                adamUpdaterId);

  if (combo->mode == AdamMode::Lamb || combo->mode == AdamMode::LambNoBias) {
    adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getLambR1SqInIndex(),
                                  lambR1SqId);
    adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getLambR2SqInIndex(),
                                  lambR2SqId);
  }

  // Optimizer parameters
  if (!combo->initLr.isConst()) {
    adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getLrInIndex(),
                                  combo->inId(AdamComboOp::getLrInIndex()));
  }
  if (!combo->initMwn.isConst()) {
    adamVarUpdOp->connectInTensor(AdamVarUpdateOp::getMwnInIndex(),
                                  combo->inId(AdamComboOp::getMwnInIndex()));
  }

  if (combo->withGradAccum) {
    adamVarUpdOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    adamVarUpdOp->setExecutionPhase({});
    adamVarUpdOp->settings.schedulePriority = 0.0;
  } else {
    graph.topoCons->transfer(combo, adamVarUpdOp);
  }

  // deleting combo op now, so that its output can be re-connected
  combo->disconnectAllInputs();
  combo->disconnectAllOutputs();
  graph.eraseOp(combo->id);

  adamVarUpdOp->connectOutTensor(AdamVarUpdateOp::getUpdatedVarOutIndex(),
                                 updatedWeightId);
  adamVarUpdOp->setup();

  return true;
}

namespace {
// Not registering this pattern, as we want it to run at a special time (after
// matmul serialization)
static AddPatternName<AdamDecompose> registerName("AdamDecompose");
} // namespace

} // namespace popart
