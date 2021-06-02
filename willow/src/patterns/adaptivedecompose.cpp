// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ces/concatce.hpp>
#include <popart/ces/flattence.hpp>
#include <popart/ces/slicece.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/adadeltaupdater.hpp>
#include <popart/op/adaptivecombo.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/rmspropupdater.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/scaledvarupdate.hpp>
#include <popart/op/slice.hpp>
#include <popart/patterns/adaptivedecompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

namespace popart {

bool AdaptiveDecompose::matches(Op *op) const {
  return op->isConvertibleTo<AdaptiveComboOp>();
}

std::vector<const Tensor *> AdaptiveDecompose::touches(Op *) const {
  return {};
}

bool AdaptiveDecompose::apply(Op *op) const {
  auto &graph = op->getGraph();

  // matches must have verified the correctness before this call
  auto combo = static_cast<AdaptiveComboOp *>(op);

  Tensor *weightGrad =
      combo->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  Tensor *weight    = combo->inTensor(VarUpdateOp::getVarToUpdateInIndex());
  Tensor *newWeight = combo->outTensor(VarUpdateOp::getUpdatedVarOutIndex());

  TensorId weightGradId    = weightGrad->id;
  TensorId weightId        = weight->id;
  TensorId updatedWeightId = newWeight->id;

  // Qualified tensor names for key tensors of the Adaptive/Lamb optimizers
  auto stepId            = reservedStepPrefix() + weightId;
  auto accumId           = reservedAccumPrefix() + weightId;
  auto accl1Id           = reservedAccl1Prefix() + weightId;
  auto accl2Id           = reservedAccl2Prefix() + weightId;
  auto accl3Id           = reservedAccl3Prefix() + weightId;
  auto adaptiveUpdaterId = reservedAdaptiveUpdaterPrefix() + weightId;

  if (weightGrad->info.dataType() != weight->info.dataType()) {
    throw error("Currently, weight and weight gradient should have the same "
                "type in AdaptiveDecompose, this is outstanding work");
  }

  auto weightInfo  = weight->info;
  auto weightShape = weightInfo.shape();

  // Accumulator
  if (combo->withGradAccum) {
    addStateTensor(graph, accumId, weightShape, combo->accumType);
  }

  if (combo->rmspropTFVariant) {
    // In TF variant of RMSProp, the accumulator buffer is initialized to ones
    // rather than zeros.
    addStateTensor(graph, accl1Id, weightShape, combo->accl1Type, 1.0);
  } else {
    addStateTensor(graph, accl1Id, weightShape, combo->accl1Type);
  }

  if (combo->mode == AdaptiveMode::CenteredRMSProp ||
      combo->mode == AdaptiveMode::AdaDelta) {
    addStateTensor(graph, accl2Id, weightShape, combo->accl2Type);
  }

  bool useMomentum = !combo->initM.isConst() || combo->initM.val() > 0.0f;
  if (useMomentum) {
    addStateTensor(graph, accl3Id, weightShape, combo->accl3Type);
  }

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
  TensorId gsId = combo->initGs.isConst()
                      ? ""
                      : combo->inId(AdaptiveComboOp::getGsInIndex());
  gradIntoAcclId = gradUnscale(
      graph, combo, combo->initGs, gsId, gradIntoAcclId, combo->withGradAccum);

  // L2 regularization
  if (combo->decayMode == WeightDecayMode::L2Regularization) {
    TensorId wdId = combo->initWd.isConst()
                        ? ""
                        : combo->inId(AdaptiveComboOp::getWdInIndex());
    gradIntoAcclId = regularizeL2(graph,
                                  combo,
                                  combo->initWd,
                                  wdId,
                                  weightId,
                                  gradIntoAcclId,
                                  combo->withGradAccum);
  }

  std::vector<Op *> beforeAccumUpdateOps;

  // Accl1
  auto accl1 = accl(
      graph,
      combo,
      accl1Id,
      gradIntoAcclId,
      combo->mode == AdaptiveMode::AdaGrad
          ? AccumulationType::DampenedAddSquare
          : AccumulationType::MovingAverageSquare,
      combo->mode == AdaptiveMode::AdaGrad ? OptimizerValue(1.0f, true)
                                           : combo->initA,
      combo->initA.isConst() ? ""
                             : combo->inId(AdaptiveComboOp::getAlphaInIndex()),
      "_accl1",
      combo->withGradAccum);
  Op *accl1Op             = accl1.first;
  TensorId updatedAccl1Id = accl1.second;
  beforeAccumUpdateOps.push_back(accl1Op);

  TensorId updatedAccl2Id;
  if (combo->mode == AdaptiveMode::CenteredRMSProp) {
    // Accl2
    auto accl2     = accl(graph,
                      combo,
                      accl2Id,
                      gradIntoAcclId,
                      AccumulationType::MovingAverage,
                      combo->initA,
                      combo->initA.isConst()
                          ? ""
                          : combo->inId(AdaptiveComboOp::getAlphaInIndex()),
                      "_accl2",
                      combo->withGradAccum);
    Op *accl2Op    = accl2.first;
    updatedAccl2Id = accl2.second;
    beforeAccumUpdateOps.push_back(accl2Op);
  }

  // Adaptive updater term
  Op *adaptiveUpdOp;
  OutIndex updaterOutIndex;
  if (combo->mode == AdaptiveMode::AdaDelta) {
    auto adaptiveUpdOpUp = std::make_unique<AdaDeltaUpdaterOp>(
        combo->initEps,
        Op::Settings(graph, combo->name() + "_adadeltaupdater"));
    adaptiveUpdOp = adaptiveUpdOpUp.get();
    transferBaseProperties(combo, adaptiveUpdOp);
    graph.moveIntoGraph(std::move(adaptiveUpdOpUp));
    updaterOutIndex = AdaDeltaUpdaterOp::getUpdaterOutIndex();

    // Gradient
    logging::pattern::trace("Connecting input {} to {} at {}",
                            gradIntoAcclId,
                            adaptiveUpdOp->str(),
                            AdaDeltaUpdaterOp::getGradInIndex());
    adaptiveUpdOp->connectInTensor(AdaDeltaUpdaterOp::getGradInIndex(),
                                   gradIntoAcclId);

    // 1st momentum
    logging::pattern::trace("Connecting input {} to {} at {}",
                            accl1Id,
                            adaptiveUpdOp->str(),
                            AdaDeltaUpdaterOp::getAccl1InIndex());
    adaptiveUpdOp->connectInTensor(AdaDeltaUpdaterOp::getAccl1InIndex(),
                                   updatedAccl1Id);

    // 2nd momentum
    logging::pattern::trace("Connecting input {} to {} at {}",
                            accl2Id,
                            adaptiveUpdOp->str(),
                            AdaDeltaUpdaterOp::getAccl2InIndex());
    adaptiveUpdOp->connectInTensor(AdaDeltaUpdaterOp::getAccl2InIndex(),
                                   accl2Id);

    // Optimizer parameters
    if (!combo->initEps.isConst()) {
      adaptiveUpdOp->connectInTensor(
          AdaDeltaUpdaterOp::getEpsInIndex(),
          combo->inId(AdaptiveComboOp::getEpsInIndex()));
    }
  } else {
    auto adaptiveUpdOpUp = std::make_unique<RMSPropUpdaterOp>(
        combo->initEps,
        combo->rmspropTFVariant,
        Op::Settings(graph, combo->name() + "_rmspropupdater"));
    adaptiveUpdOp = adaptiveUpdOpUp.get();
    transferBaseProperties(combo, adaptiveUpdOp);
    graph.moveIntoGraph(std::move(adaptiveUpdOpUp));
    updaterOutIndex = RMSPropUpdaterOp::getUpdaterOutIndex();

    // Gradient
    logging::pattern::trace("Connecting input {} to {} at {}",
                            gradIntoAcclId,
                            adaptiveUpdOp->str(),
                            RMSPropUpdaterOp::getGradInIndex());
    adaptiveUpdOp->connectInTensor(RMSPropUpdaterOp::getGradInIndex(),
                                   gradIntoAcclId);

    logging::pattern::trace("Connecting input {} to {} at {}",
                            updatedAccl1Id,
                            adaptiveUpdOp->str(),
                            RMSPropUpdaterOp::getAccl1InIndex());
    adaptiveUpdOp->connectInTensor(RMSPropUpdaterOp::getAccl1InIndex(),
                                   updatedAccl1Id);

    if (combo->mode == AdaptiveMode::CenteredRMSProp) {
      logging::pattern::trace("Connecting input {} to {} at {}",
                              updatedAccl2Id,
                              adaptiveUpdOp->str(),
                              RMSPropUpdaterOp::getAccl2InIndex());
      adaptiveUpdOp->connectInTensor(RMSPropUpdaterOp::getAccl2InIndex(),
                                     updatedAccl2Id);
    }

    // Optimizer parameters
    if (!combo->initEps.isConst()) {
      adaptiveUpdOp->connectInTensor(
          AdaDeltaUpdaterOp::getEpsInIndex(),
          combo->inId(AdaptiveComboOp::getEpsInIndex()));
    }
  }

  // The accumulator updater
  if (combo->withGradAccum) {
    beforeAccumUpdateOps.push_back(adaptiveUpdOp);
    accumUpdate(graph, combo, beforeAccumUpdateOps, accumId);
  }

  // Updater term
  adaptiveUpdOp->createAndConnectOutTensor(updaterOutIndex, adaptiveUpdaterId);

  adaptiveUpdOp->setup();
  if (combo->withGradAccum) {
    adaptiveUpdOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    adaptiveUpdOp->setExecutionPhase({});
    adaptiveUpdOp->settings.schedulePriority = 0.0;
  }

  if (combo->mode == AdaptiveMode::AdaDelta) {
    // Accl2
    auto accl2     = accl(graph,
                      combo,
                      accl2Id,
                      adaptiveUpdaterId,
                      AccumulationType::MovingAverageSquare,
                      combo->initA,
                      combo->initA.isConst()
                          ? ""
                          : combo->inId(AdaptiveComboOp::getAlphaInIndex()),
                      "_accl2",
                      combo->withGradAccum);
    updatedAccl2Id = accl2.second;
  }

  if (useMomentum) {
    // Accl3
    if (combo->rmspropTFVariant) {
      // TF variant of RMSProp accumulates learning rate in the momentum buffer
      // so we scale the updater term by learning rate before creating the
      // actual momentum accumulator.
      Op *updaterScaleOp;

      if (combo->initLr.isConst()) {
        auto updaterScaleOpUp = std::make_unique<ScaleOp>(
            Onnx::AiGraphcore::OpSet1::Scale,
            combo->initLr.val(),
            Op::Settings(graph, combo->name() + "_tfrmspropupdatescaler"));
        updaterScaleOp = updaterScaleOpUp.get();
        transferBaseProperties(combo, updaterScaleOp);
        graph.moveIntoGraph(std::move(updaterScaleOpUp));

        logging::pattern::trace("Connecting input {} to {} at {}",
                                adaptiveUpdaterId,
                                updaterScaleOp->str(),
                                ScaleOp::getInIndex());
        updaterScaleOp->connectInTensor(ScaleOp::getInIndex(),
                                        adaptiveUpdaterId);

        adaptiveUpdaterId =
            graph.getIr().createIntermediateTensorId(adaptiveUpdaterId);
        updaterScaleOp->createAndConnectOutTensor(ScaleOp::getOutIndex(),
                                                  adaptiveUpdaterId);
      } else {
        auto updaterScaleOpUp = std::make_unique<MulOp>(
            Onnx::AiOnnx::OpSet11::Mul,
            Op::Settings(graph, combo->name() + "_tfrmspropupdatescaler"));
        updaterScaleOp = updaterScaleOpUp.get();
        transferBaseProperties(combo, updaterScaleOp);
        graph.moveIntoGraph(std::move(updaterScaleOpUp));

        auto lrTensorId = combo->inId(AdaptiveComboOp::getLrInIndex());
        logging::pattern::trace("Connecting input {} to {} at {}",
                                lrTensorId,
                                updaterScaleOp->str(),
                                MulOp::getArg0InIndex());
        updaterScaleOp->connectInTensor(MulOp::getArg0InIndex(), lrTensorId);

        logging::pattern::trace("Connecting input {} to {} at {}",
                                adaptiveUpdaterId,
                                updaterScaleOp->str(),
                                MulOp::getArg1InIndex());
        updaterScaleOp->connectInTensor(MulOp::getArg1InIndex(),
                                        adaptiveUpdaterId);

        adaptiveUpdaterId =
            graph.getIr().createIntermediateTensorId(adaptiveUpdaterId);
        updaterScaleOp->createAndConnectOutTensor(MulOp::getOutIndex(),
                                                  adaptiveUpdaterId);
      }

      updaterScaleOp->setup();
      if (combo->withGradAccum) {
        updaterScaleOp->settings.executionContext =
            ExecutionContext::AccumulateOuterFragment;
        updaterScaleOp->setExecutionPhase({});
        updaterScaleOp->settings.schedulePriority = 0.0;
      }
    }

    auto accl3              = accl(graph,
                      combo,
                      accl3Id,
                      adaptiveUpdaterId,
                      AccumulationType::DecayAdd,
                      combo->initM,
                      combo->initM.isConst()
                          ? ""
                          : combo->inId(AdaptiveComboOp::getMomentumInIndex()),
                      "_accl3",
                      combo->withGradAccum);
    TensorId updatedAccl3Id = accl3.second;
    // Var update uses momentum updater instead
    adaptiveUpdaterId = updatedAccl3Id;
  }

  // Var update
  auto scaledVarUpdOpUp = std::make_unique<ScaledVarUpdateOp>(
      combo->initLr,
      combo->decayMode == WeightDecayMode::Decay ? combo->initWd
                                                 : OptimizerValue(0.0f, true),
      combo->rmspropTFVariant && useMomentum,
      Op::Settings(graph, combo->name() + "_var_update"));
  auto scaledVarUpdOp = scaledVarUpdOpUp.get();
  transferBaseProperties(combo, scaledVarUpdOp);
  graph.moveIntoGraph(std::move(scaledVarUpdOpUp));

  // Weight
  logging::pattern::trace("Connecting input {} to {} at {}",
                          weightId,
                          scaledVarUpdOp->str(),
                          ScaledVarUpdateOp::getVarToUpdateInIndex());
  scaledVarUpdOp->connectInTensor(ScaledVarUpdateOp::getVarToUpdateInIndex(),
                                  weightId);

  // Updater
  logging::pattern::trace("Connecting input {} to {} at {}",
                          accl1Id,
                          scaledVarUpdOp->str(),
                          ScaledVarUpdateOp::getUpdaterInIndex());
  scaledVarUpdOp->connectInTensor(ScaledVarUpdateOp::getUpdaterInIndex(),
                                  adaptiveUpdaterId);

  // Optimizer parameters
  if (!combo->initLr.isConst() && !(combo->rmspropTFVariant && useMomentum)) {
    scaledVarUpdOp->connectInTensor(
        ScaledVarUpdateOp::getLrInIndex(),
        combo->inId(AdaptiveComboOp::getLrInIndex()));
  }
  if (!combo->initWd.isConst()) {
    scaledVarUpdOp->connectInTensor(
        ScaledVarUpdateOp::getWdInIndex(),
        combo->inId(AdaptiveComboOp::getWdInIndex()));
  }

  if (combo->withGradAccum) {
    scaledVarUpdOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    scaledVarUpdOp->setExecutionPhase({});
    scaledVarUpdOp->settings.schedulePriority = 0.0;
  } else {
    graph.topoCons->transfer(combo, scaledVarUpdOp);
  }

  // deleting combo op now, so that its output can be re-connected
  combo->disconnectAllInputs();
  combo->disconnectAllOutputs();
  graph.eraseOp(combo->id);

  scaledVarUpdOp->connectOutTensor(ScaledVarUpdateOp::getUpdatedVarOutIndex(),
                                   updatedWeightId);

  scaledVarUpdOp->setup();

  return true;
}

namespace {
// Not registering this pattern, as we want it to run at a special time (after
// matmul serialization)
static AddPatternName<AdaptiveDecompose> registerName("AdaptiveDecompose");
} // namespace

} // namespace popart
