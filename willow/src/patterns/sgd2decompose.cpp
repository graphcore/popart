// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/sgd1nesterov.hpp>
#include <popart/op/sgd2acclupdate.hpp>
#include <popart/op/sgd2combo.hpp>
#include <popart/op/sgd2varupdate.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/sgd2decompose.hpp>
#include <popart/topocons.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/operators.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensornames.hpp"
#include "popart/tensors.hpp"
#include "popart/variablesettings.hpp"

namespace popart {

bool SGD2Decompose::matches(Op *op) const {
  return op->isConvertibleTo<SGD2ComboOp>();
}

std::vector<const Tensor *> SGD2Decompose::touches(Op *) const { return {}; }

bool SGD2Decompose::apply(Op *op) const {
  auto &graph = op->getGraph();

  // Matches must have verified the correctness before this call
  auto combo = static_cast<SGD2ComboOp *>(op);

  Tensor *weightGrad =
      combo->inTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  Tensor *weight    = combo->inTensor(VarUpdateOp::getVarToUpdateInIndex());
  Tensor *newWeight = combo->outTensor(VarUpdateOp::getUpdatedVarOutIndex());

  VariableSettings varset = weight->getVariableSettings();

  TensorId weightGradId    = weightGrad->id;
  TensorId weightId        = weight->id;
  TensorId updatedWeightId = newWeight->id;

  // Qualified tensor names for the state tensors of the SGD optimizer. We will
  // create these tensors.
  // Note, we must use Accl1, not Accl, as Accl is a special Accl+Accum tensor
  // used by SGD1.
  auto accumId = reservedAccumPrefix() + weightId;
  auto accl1Id = reservedAccl1Prefix() + weightId;

  if (weightGrad->info.dataType() != weight->info.dataType()) {
    throw error("Currently, weight and weight gradient should have the same "
                "type in SGD2Decompose, this is outstanding work");
  }

  auto weightInfo  = weight->info;
  auto weightShape = weightInfo.shape();

  // Accumulator
  if (combo->withGradAccum) {
    addStateTensor(
        graph,
        accumId,
        weightShape,
        combo->accumType,
        VariableSettings(
            graph.getIr().getSessionOptions().getGlobalReplicationFactor()));
  }

  // 1st momentum (accl1)
  addStateTensor(graph, accl1Id, weightShape, combo->accl1Type, varset);

  // If doing gradient accumulation with AccumReduce reduction, the gradients
  // will first go to an op that updates the accumulator, then this accumulator
  // goes to the accl updater op.
  TensorId gradIntoAcclId  = weightGradId;
  TensorId gradIntoAccumId = weightGradId;

  if (combo->reductionType == OptimizerReductionType::GradReduce) {
    TensorId reducedId = gradReduce(graph, combo, weightId, weightGradId);
    gradIntoAcclId     = reducedId;
    gradIntoAccumId    = reducedId;
  }

  // Gradient accumulation
  if (combo->withGradAccum) {
    gradIntoAcclId =
        gradAccum(graph,
                  combo,
                  weightId,
                  accumId,
                  gradIntoAccumId,
                  combo->reductionType == OptimizerReductionType::AccumReduce);
  }

  // Remaining ops run after the gradient accumulation loop (if enabled)

  TensorId updatedAcc1lId =
      acclUpdate(graph, combo, gradIntoAcclId, accl1Id, weightId);

  // Zero the gradient accumulator after updating the 1st momentum term
  // ready for next step
  if (combo->withGradAccum && !runningMeanReduction(graph)) {
    const auto acclOp = graph.getTensors().get(updatedAcc1lId)->getProducer();
    zeroAccumulator(graph, combo, {acclOp}, accumId);
  }

  // If enable nesterov momentum, update gradient
  if (combo->nesterov) {
    updatedAcc1lId = nesterovGradUpdate(
        graph, combo, gradIntoAcclId, weightId, updatedAcc1lId);
  }

  varUpdateAndEraseCombo(
      graph, combo, weightId, updatedAcc1lId, updatedWeightId);

  return true;
}

TensorId SGD2Decompose::acclUpdate(Graph &graph,
                                   const SGD2ComboOp *combo,
                                   const TensorId &gradIntoAcclId,
                                   const TensorId &accl1Id,
                                   const TensorId &weightId) const {
  graph.getIr().addAdditionalModelProtoTensor(accl1Id);

  //  We will update accl1Id in two steps.
  //
  //  First, through an SGD2PartialAcclUpdateOp, which is equivalent to an
  //  SGD1AcclUpdateOp. This performs
  //    v <- smm1 * v + swd1 * w
  //  Second, we perform an AccumulateOp on the updated v:
  //    v <- v + dpsf1 * g
  //
  //  Together, this implements the SGD update equation for v. See the equation
  //  derivations in optimizer.hpp for how this equation is derived, and the
  //  definitions of the compound scalars smm1, swd1, dpsf1.

  // TensorIds to use as the outputs of the two ops.
  const auto accl1Id_0 = graph.getIr().createIntermediateTensorId(accl1Id);
  const auto accl1Id_1 = graph.getIr().createIntermediateTensorId(accl1Id_0);

  // SGD2PartialAcclUpdateOp
  //
  // Inputs
  // (1) accl1Id [the updated var] (not yet modified this optimiser step)
  // (2) weightId [the updater] (not yet modified this optimiser step)
  // (3) Smm1 (only if not const)
  // (4) Wdsf1 (only if not const)
  //
  // Outputs
  // (5) accl1Id_0
  {
    auto acclUpdate0 = graph.createOp<SGD2PartialAcclUpdateOp>(
        combo->initSmm1,
        combo->initSwd1,
        Op::Settings(graph,
                     combo->name() + "_accl_update_0_prev_v_w",
                     combo->settings.debugInfoId));
    transferBaseProperties(combo, acclUpdate0);

    // (1)
    acclUpdate0->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(), accl1Id);

    // (2)
    acclUpdate0->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                                 weightId);
    // (3)
    if (!combo->initSmm1.isConst()) {
      acclUpdate0->connectInTensor(SGD2PartialAcclUpdateOp::getSmm1InIndex(),
                                   combo->inId(SGD2ComboOp::getSmm1InIndex()));
    }

    // (4)
    if (!combo->initSwd1.isConst()) {
      acclUpdate0->connectInTensor(SGD2PartialAcclUpdateOp::getSwd1InIndex(),
                                   combo->inId(SGD2ComboOp::getSwd1InIndex()));
    }

    // (5)
    acclUpdate0->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                           accl1Id_0);
    acclUpdate0->setup();

    if (combo->withGradAccum) {
      acclUpdate0->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
      acclUpdate0->setExecutionPhase({});
      acclUpdate0->settings.schedulePriority = 0.0;
    }
  }

  // AccumulateOp
  //
  // Parameters
  // AccumulationType::DampenedAdd
  //
  // Inputs
  // (1) accl1Id_0 [the updated var] (output of previous op)
  // (2) gradIntoAcclId [the updater] the grads for this optimiser step
  // (3) Dpsf1 [the factor] (only if not const)
  //
  // Outputs
  // (4) accl1Id_1
  {
    auto acclUpdate1 = graph.createOp<AccumulateOp>(
        AccumulationType::DampenedAdd,
        combo->initDpsf1,
        Op::Settings(graph,
                     combo->name() + "_accl_update_1_grads",
                     combo->settings.debugInfoId));
    transferBaseProperties(combo, acclUpdate1);

    // (1)
    acclUpdate1->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                                 accl1Id_0);

    // (2)
    acclUpdate1->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                                 gradIntoAcclId);

    // (3)
    if (!combo->initDpsf1.isConst()) {
      acclUpdate1->connectInTensor(AccumulateOp::getFactorInIndex(),
                                   combo->inId(SGD2ComboOp::getDpsf1InIndex()));
    }

    // (4)
    acclUpdate1->createAndConnectOutTensor(
        AccumulateOp::getUpdatedVarOutIndex(), accl1Id_1);

    acclUpdate1->setup();

    if (combo->withGradAccum) {
      acclUpdate1->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
      acclUpdate1->setExecutionPhase({});
      acclUpdate1->settings.schedulePriority = 0.0;
    }
  }

  return accl1Id_1;
}

TensorId
SGD2Decompose::nesterovGradUpdate(Graph &graph,
                                  const SGD2ComboOp *combo,
                                  const TensorId &gradIntoAcclId,
                                  const TensorId &weightId,
                                  const TensorId &updatedAcc1lId) const {
  // TensorIds to use as the outputs of the AccumulateOp.
  const auto nesterovGradUpdatedId = weightId + "_nesterovGrad";

  // Get inverse loss scale tensor(Dpsf1 * Ndsf = 1 / ls)
  TensorId sgdInverselossScale = "";
  if (!combo->initDpsf1.isConst()) {
    auto mulOp = graph.createOp<MulOp>(
        Onnx::AiOnnx::OpSet6::Mul,
        Op::Settings(
            graph, combo->name() + "_nesterov_0", combo->settings.debugInfoId));
    transferBaseProperties(combo, mulOp);
    mulOp->connectInTensor(MulOp::getArg0InIndex(),
                           combo->inId(SGD2ComboOp::getDpsf1InIndex()));
    mulOp->connectInTensor(MulOp::getArg1InIndex(),
                           combo->inId(SGD2ComboOp::getNdsfInIndex()));
    mulOp->createAndConnectOutTensor(MulOp::getOutIndex(),
                                     weightId + "_sgdInverseLossScale");
    mulOp->setup();
    if (combo->withGradAccum) {
      mulOp->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
      mulOp->settings.schedulePriority = 0.0;
    }
    sgdInverselossScale = mulOp->outTensor(MulOp::getOutIndex())->id;
  }

  // SGD1Nesterov Op
  //
  // g_out = ngsf * (1 / ls * g + wd * w) + mm * v
  //
  // Inputs
  // (1) g - Gradient
  // (2) w - Weight
  // (3) v - Velocity
  // (4) 1 / (ls * rf) - Inverse loss scale
  // (5) wd - Weight decay
  // (6) ngsf - Nesterov gradient scale factor(= vs)
  // (7) mm - Momentum
  //
  // Outputs
  // (8) g_out - Nesterov updated gradient
  auto nesterovOp = graph.createOp<SGD1NesterovOp>(
      Onnx::CustomOperators::SGD1Nesterov,
      combo->initNdsf.val() * combo->initDpsf1.val(),
      combo->initWd.val(),
      combo->initNgsf.val(),
      combo->initMm.val(),
      Op::Settings(
          graph, combo->name() + "_nesterov_1", combo->settings.debugInfoId));
  transferBaseProperties(combo, nesterovOp);

  // (1)
  nesterovOp->connectInTensor(SGD1NesterovOp::getGradInIndex(), gradIntoAcclId);
  // (2)
  nesterovOp->connectInTensor(SGD1NesterovOp::getWeightInIndex(), weightId);

  // (3)
  nesterovOp->connectInTensor(SGD1NesterovOp::getVelocityInIndex(),
                              updatedAcc1lId);

  // (4)
  if (!combo->initNdsf.isConst()) {
    nesterovOp->connectInTensor(SGD1NesterovOp::getInverseLossScaleInIndex(),
                                sgdInverselossScale);
  }

  // (5)
  if (!combo->initWd.isConst()) {
    nesterovOp->connectInTensor(SGD1NesterovOp::getWdInIndex(),
                                combo->inId(SGD2ComboOp::getWdInIndex()));
  }

  // (6)
  if (!combo->initNgsf.isConst()) {
    nesterovOp->connectInTensor(SGD1NesterovOp::getNgsfInIndex(),
                                combo->inId(SGD2ComboOp::getNgsfInIndex()));
  }

  // (7)
  if (!combo->initMm.isConst()) {
    nesterovOp->connectInTensor(SGD1NesterovOp::getMmInIndex(),
                                combo->inId(SGD2ComboOp::getMmInIndex()));
  }

  // (8)
  nesterovOp->createAndConnectOutTensor(SGD1NesterovOp::getOutIndex(),
                                        nesterovGradUpdatedId);

  nesterovOp->setup();

  if (combo->withGradAccum) {
    nesterovOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    nesterovOp->settings.schedulePriority = 0.0;
  }

  return nesterovGradUpdatedId;
}

void SGD2Decompose::varUpdateAndEraseCombo(
    Graph &graph,
    SGD2ComboOp *combo,
    const TensorId &weightId,
    const TensorId &updatedAcc1lId,
    const TensorId &updatedWeightId) const {
  // We update the weight using an SGD2VarUpdateOp, which is exactly equivalent
  // to an SGD1VarUpdateOp. That is, we perform:
  //   w <- w - slr1 * v
  //
  // See the equation derivations in optimizer.hpp for how this is derived, and
  // the definition of the compound scalar slr1.

  // SGD2VarUpdateOp
  //
  // Inputs
  // (1) weightId [the updated var]
  // (2) updatedAcc1lId [the updater] accl updated this optimiser step
  // (3) slr1 (only if not const)
  //
  // Outputs
  // (4) updatedWeightId

  auto sgd2VarUpdate = graph.createOp<SGD2VarUpdateOp>(
      combo->initSlr1,
      Op::Settings(
          graph, combo->name() + "_var_update", combo->settings.debugInfoId));
  transferBaseProperties(combo, sgd2VarUpdate);

  // (1)
  sgd2VarUpdate->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(),
                                 weightId);
  // (2)
  sgd2VarUpdate->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                                 updatedAcc1lId);
  // (3)
  if (!combo->initSlr1.isConst()) {
    sgd2VarUpdate->connectInTensor(SGD2VarUpdateOp::getSlr1InIndex(),
                                   combo->inId(SGD2ComboOp::getSlr1InIndex()));
  }

  if (combo->withGradAccum) {
    sgd2VarUpdate->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    sgd2VarUpdate->setExecutionPhase({});
    sgd2VarUpdate->settings.schedulePriority = 0.0;
  } else {
    graph.topoCons->transfer(combo, sgd2VarUpdate);
  }

  // Deleting combo op now, so that its output can be re-connected.
  combo->disconnectAllInputs();
  combo->disconnectAllOutputs();
  graph.eraseOp(combo->id);

  // (4)
  sgd2VarUpdate->connectOutTensor(SGD2VarUpdateOp::getUpdatedVarOutIndex(),
                                  updatedWeightId);
  sgd2VarUpdate->setup();
}

namespace {
// Do not register this pattern with PreAliasPatternManager, as we want to
// manually control when to run it in a separate pass. We still want to register
// the pattern with PatternNames though.
static AddPatternName<SGD2Decompose> registerName("SGD2Decompose");
} // namespace

} // namespace popart
