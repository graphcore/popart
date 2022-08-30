// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <limits>
#include <map>
#include <memory>
#include <onnxutil.hpp>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/accumulatorzero.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/scaledadd.hpp>
#include <popart/patterns/optimizerdecompose.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

#include "popart/commgroup.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/half.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/operators.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensornames.hpp"
#include "popart/tensors.hpp"
#include "popart/variablesettings.hpp"

namespace popart {

template <typename T>
void OptimizerDecompose::addStateTensor(Graph &graph,
                                        const TensorId &tensorId,
                                        const TensorInfo info,
                                        const VariableSettings &varset,
                                        float initValue) const {
  auto &ir = graph.getIr();
  if (ir.tensorExistsInInitialisers(tensorId)) {
    auto tp = onnxutil::getTensorProto(ir.getModel(), tensorId);
    graph.getTensors().addVarInit(tensorId, &tp, varset);
  } else {
    // adjust number of elements w.r.t. initialization count
    auto nelms_base = info.nelms();
    auto nelms_repl =
        varset.getGroupCount(ir.getSessionOptions().replicatedGraphCount);

    std::vector<T> d(nelms_base * nelms_repl, static_cast<T>(initValue));

    // When there is non nontrivial number (1) of groups prepend the
    // groups number into shape of this state tensor.
    if (nelms_repl > 1) {
      Shape tensorShape = info.shape();
      Shape fullShape   = {nelms_repl};
      fullShape.insert(fullShape.end(), tensorShape.begin(), tensorShape.end());
      TensorInfo infoState;
      infoState.set(info.dataType(), fullShape);
      graph.getTensors().addVarInit(tensorId, infoState, d.data(), varset);
    } else {
      graph.getTensors().addVarInit(tensorId, info, d.data(), varset);
    }
  }
}

template void
OptimizerDecompose::addStateTensor<float>(Graph &graph,
                                          const TensorId &tensorId,
                                          const TensorInfo info,
                                          const VariableSettings &varset,
                                          float initValue) const;

template void
OptimizerDecompose::addStateTensor<float16_t>(Graph &graph,
                                              const TensorId &tensorId,
                                              const TensorInfo info,
                                              const VariableSettings &varset,
                                              float initValue) const;

TensorInfo OptimizerDecompose::addStateTensor(Graph &graph,
                                              const TensorId &tensorId,
                                              const Shape &shape,
                                              const DataType &type,
                                              const VariableSettings &varset,
                                              float initValue) const {
  auto info = TensorInfo(type, shape);
  switch (type) {
  case DataType::FLOAT:
    addStateTensor<float>(graph, tensorId, info, varset, initValue);
    break;
  case DataType::FLOAT16:
    addStateTensor<float16_t>(graph, tensorId, info, varset, initValue);
    break;
  default:
    throw error("Unsupported data type for tensor {}, "
                "currently only FLOAT16 and FLOAT are supported",
                tensorId);
  }
  return info;
}

std::pair<Op *, TensorId> OptimizerDecompose::accl(Graph &graph,
                                                   Op *combo,
                                                   TensorId acclId,
                                                   TensorId gradIntoAcclId,
                                                   AccumulationType type,
                                                   OptimizerValue value,
                                                   TensorId valueTensorId,
                                                   std::string acclName,
                                                   bool gradAccum) const {
  auto acclOpUp = std::make_unique<AccumulateOp>(
      type,
      value,
      Op::Settings(
          graph, combo->name() + acclName, combo->settings.debugInfoId));
  auto acclOp = acclOpUp.get();
  transferBaseProperties(combo, acclOp);
  graph.moveIntoGraph(std::move(acclOpUp));

  logging::pattern::trace("Connecting input {} to {} at {}",
                          acclId,
                          acclOp->str(),
                          VarUpdateOp::getVarToUpdateInIndex());
  acclOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(), acclId);

  logging::pattern::trace("Connecting input {} to {} at {}",
                          gradIntoAcclId,
                          acclOp->str(),
                          VarUpdateWithUpdaterOp::getUpdaterInIndex());
  acclOp->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                          gradIntoAcclId);

  // The updated accl
  TensorId updatedAcclId = graph.getIr().createIntermediateTensorId(acclId);

  acclOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                    updatedAcclId);

  if (!value.isConst()) {
    acclOp->connectInTensor(AccumulateOp::getFactorInIndex(), valueTensorId);
  }

  acclOp->setup();
  if (gradAccum) {
    acclOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    acclOp->setExecutionPhase({});
    acclOp->settings.schedulePriority = 0.0;
  }
  graph.getIr().addAdditionalModelProtoTensor(acclId);
  return {acclOp, updatedAcclId};
}

TensorId OptimizerDecompose::getCounterId(Op *combo) const {
  // Returns a TensorId such that there will only be 1 counter per VGraph per
  // PipelineStage. OptimizerDecompose::findOrCreateRunningMeanCounter
  // checks to see if the tensor already exists before creating it.
  std::stringstream id;
  id << reservedCounterPrefix();
  if (combo->hasVirtualGraphId()) {
    id << "_VGraph" << combo->getVirtualGraphId();
  }
  if (combo->hasPipelineStage()) {
    id << "_PStage" << combo->getPipelineStage();
  }
  return id.str();
}

std::pair<Op *, TensorId>
OptimizerDecompose::counterIncrement(Graph &graph,
                                     Op *combo,
                                     TensorId counterId) const {
  TensorInfo gradInfo(DataType::FLOAT, {});
  std::vector<float> gradData(gradInfo.nelms(), 1);
  const auto &increment = graph.getIr().createIntermediateTensorId("one");
  graph.getTensors().addConstInit(increment, gradInfo, gradData.data());

  TensorId updatedCounterId =
      graph.getIr().createIntermediateTensorId(counterId);

  auto counterIncrementOp = graph.createConnectedOp<AccumulateOp>(
      {{VarUpdateOp::getVarToUpdateInIndex(), counterId},
       {VarUpdateWithUpdaterOp::getUpdaterInIndex(), increment}},
      {{VarUpdateOp::getUpdatedVarOutIndex(), updatedCounterId}},
      AccumulationType::Add,
      OptimizerValue(1.0f),
      Op::Settings(graph,
                   combo->name() + "_counterIncrement",
                   combo->settings.debugInfoId));

  transferBaseProperties(combo, counterIncrementOp);

  return {counterIncrementOp, updatedCounterId};
}

std::pair<Op *, TensorId>
OptimizerDecompose::counterReset(Graph &graph,
                                 Op *combo,
                                 TensorId counterId) const {
  TensorId resetCounterId = graph.getIr().createIntermediateTensorId(counterId);

  auto counterResetOp = graph.createConnectedOp<AccumulatorZeroOp>(
      {{AccumulatorZeroOp::getVarToUpdateInIndex(), counterId}},
      {{AccumulatorZeroOp::getUpdatedVarOutIndex(), resetCounterId}},
      Op::Settings(
          graph, combo->name() + "_counterReset", combo->settings.debugInfoId));

  transferBaseProperties(combo, counterResetOp);
  counterResetOp->settings.executionContext =
      ExecutionContext::AccumulateOuterFragment;
  counterResetOp->setExecutionPhase({});
  counterResetOp->settings.schedulePriority = 0.0;

  return {counterResetOp, resetCounterId};
}

std::pair<Op *, TensorId>
OptimizerDecompose::findOrCreateRunningMeanCounter(Graph &graph,
                                                   Op *combo) const {
  TensorId counterId     = getCounterId(combo);
  Op *counterIncrementOp = nullptr;
  if (graph.getTensors().contains(counterId)) {
    for (auto cons : graph.getTensors().get(counterId)->consumers.getOps()) {
      if (cons->isConvertibleTo<AccumulateOp>() &&
          cons->inId(AccumulateOp::getVarToUpdateInIndex()) == counterId) {
        counterIncrementOp = cons;
        break;
      }
    }
    if (!counterIncrementOp) {
      throw internal_error("OptimiserDecompose could not find the AccumulateOp "
                           "that increments running mean counter {}",
                           counterId);
    }
  } else {
    addStateTensor<float>(
        graph, counterId, TensorInfo(DataType::FLOAT, {}), VariableSettings());
    // Note we do not add Counter__ to AdditionalModelProtoTensors, in the
    // current implementation where it is guaranteed that this is zero after one
    // run.

    auto op_tid        = counterIncrement(graph, combo, counterId);
    counterIncrementOp = op_tid.first;

    counterReset(graph, combo, op_tid.second);
  }
  return {counterIncrementOp, counterId};
}

bool OptimizerDecompose::runningMeanReduction(Graph &graph) const {
  return graph.getIr()
                 .getSessionOptions()
                 .accumulationAndReplicationReductionType ==
             ReductionType::Mean &&
         graph.getIr()
                 .getSessionOptions()
                 .meanAccumulationAndReplicationReductionStrategy ==
             MeanReductionStrategy::Running;
}

TensorId OptimizerDecompose::gradAccum(Graph &graph,
                                       Op *combo,
                                       TensorId weightId,
                                       TensorId accumId,
                                       TensorId gradIntoAccumId,
                                       bool accumReduce,
                                       TensorId outputId) const {
  bool runningAccum = runningMeanReduction(graph);

  TensorId gradIntoAcclId;
  auto accumOpUp = std::make_unique<AccumulateOp>(
      runningAccum ? AccumulationType::Mean : AccumulationType::Add,
      OptimizerValue(1.0f, !runningAccum),
      Op::Settings(
          graph, combo->name() + "_accumulate", combo->settings.debugInfoId));
  auto accumOp = accumOpUp.get();
  transferBaseProperties(combo, accumOp);
  graph.moveIntoGraph(std::move(accumOpUp));

  // The combo op is a VarUpdateOp. delayVarUpdates makes it so all VarUpdateOps
  // have negative infinity schedule priority. The transferBaseProperties call
  // above subsequently causes the gradient accumulation AccumulateOps to have
  // this schedule priority too. The earlyGradientAccumulation option tells
  // Popart to manually override this back to the default.
  //
  // If we do not have explicit main loops, the existence of the
  // AccumulateOuterFragment causes the scheduler to "pull" the AccumulateOps to
  // the end of the schedule, so we must go further and set the priority to max.
  if (graph.getIr().getSessionOptions().shouldDelayVarUpdates() &&
      graph.getIr()
          .getSessionOptions()
          .scheduleNonWeightUpdateGradientConsumersEarly) {
    accumOp->settings.schedulePriority = std::numeric_limits<double>::max();
  }

  logging::pattern::trace("Connecting input {} to {} at {}",
                          accumId,
                          accumOp->str(),
                          VarUpdateOp::getVarToUpdateInIndex());
  accumOp->connectInTensor(VarUpdateOp::getVarToUpdateInIndex(), accumId);

  logging::pattern::trace("Connecting input {} to {} at {}",
                          gradIntoAccumId,
                          accumOp->str(),
                          VarUpdateWithUpdaterOp::getUpdaterInIndex());
  accumOp->connectInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex(),
                           gradIntoAccumId);

  if (runningAccum) {
    auto op_id = findOrCreateRunningMeanCounter(graph, combo);
    accumOp->connectInTensor(AccumulateOp::getFactorInIndex(), op_id.second);
    graph.topoCons->insert(accumOp, op_id.first);
  }

  // The updated accumulator
  TensorId updatedAccumId =
      !outputId.empty() && !accumReduce
          ? outputId
          : graph.getIr().createIntermediateTensorId(accumId);

  accumOp->createAndConnectOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
                                     updatedAccumId);

  accumOp->setup();
  graph.topoCons->transfer(combo, accumOp);
  // Note we do not add Accum__ to AdditionalModelProtoTensors,  in the current
  // implementation where it is guaranteed that this is zero after one run.

  if (accumReduce) {
    bool runningReplica = runningMeanReduction(graph);
    Tensor *t           = graph.getTensors().get(weightId);
    CommGroup cg        = t->getVariableSettings().getSharedVariableDomain();
    auto reduceOpUp     = std::make_unique<ReplicatedAllReduceInplaceOp>(
        Onnx::CustomOperators::ReplicatedAllReduceInplace,
        runningReplica ? CollectiveOperator::Mean : CollectiveOperator::Add,
        cg,
        Op::Settings(
            graph, combo->name() + "_reduce", combo->settings.debugInfoId));
    auto reduceOp = reduceOpUp.get();
    transferBaseProperties(combo, reduceOp);
    graph.moveIntoGraph(std::move(reduceOpUp));

    logging::pattern::trace("Connecting input {} to {} at {}",
                            updatedAccumId,
                            reduceOp->str(),
                            ReplicatedAllReduceInplaceOp::getInIndex());
    reduceOp->connectInTensor(ReplicatedAllReduceInplaceOp::getInIndex(),
                              updatedAccumId);

    // The updated, reduced accumulator
    gradIntoAcclId = !outputId.empty()
                         ? outputId
                         : graph.getIr().createIntermediateTensorId(accumId);

    reduceOp->createAndConnectOutTensor(
        ReplicatedAllReduceInplaceOp::getOutIndex(), gradIntoAcclId);

    reduceOp->setup();
    reduceOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    reduceOp->setExecutionPhase({});
    reduceOp->settings.schedulePriority = 0.0;
  } else {
    // No replicated accumulator reduction
    gradIntoAcclId = updatedAccumId;
  }
  return gradIntoAcclId;
}

Op *OptimizerDecompose::zeroAccumulator(Graph &graph,
                                        Op *combo,
                                        std::vector<Op *> beforeOps,
                                        TensorId accumId) const {
  auto accumZeroOpUp = std::make_unique<AccumulatorZeroOp>(Op::Settings(
      graph, combo->name() + "_accumupdate", combo->settings.debugInfoId));
  auto accumZeroOp   = accumZeroOpUp.get();
  transferBaseProperties(combo, accumZeroOp);
  graph.moveIntoGraph(std::move(accumZeroOpUp));

  logging::pattern::trace("Connecting input {} to {} at {}",
                          accumId,
                          accumZeroOp->str(),
                          AccumulatorZeroOp::getVarToUpdateInIndex());
  accumZeroOp->connectInTensor(AccumulatorZeroOp::getVarToUpdateInIndex(),
                               accumId);

  TensorId updatedAccumId = reservedUpdatedVarPrefix() + accumId;

  accumZeroOp->createAndConnectOutTensor(
      AccumulatorZeroOp::getUpdatedVarOutIndex(), updatedAccumId);

  accumZeroOp->setup();

  accumZeroOp->settings.executionContext =
      ExecutionContext::AccumulateOuterFragment;
  accumZeroOp->setExecutionPhase({});
  accumZeroOp->settings.schedulePriority = 0.0;

  for (Op *beforeOp : beforeOps) {
    graph.topoCons->insert(beforeOp, accumZeroOp);
  }
  return accumZeroOp;
}

TensorId OptimizerDecompose::gradReduce(Graph &graph,
                                        Op *combo,
                                        TensorId weightId,
                                        TensorId weightGradId,
                                        TensorId outputId) const {
  bool runningMean = runningMeanReduction(graph);
  Tensor *t        = graph.getTensors().get(weightId);
  CommGroup cg     = t->getVariableSettings().getSharedVariableDomain();
  auto reduceOpUp  = std::make_unique<ReplicatedAllReduceOp>(
      Onnx::CustomOperators::ReplicatedAllReduce,
      runningMean ? CollectiveOperator::Mean : CollectiveOperator::Add,
      cg,
      Op::Settings(
          graph, combo->name() + "_reduce", combo->settings.debugInfoId));
  auto reduceOp = reduceOpUp.get();
  transferBaseProperties(combo, reduceOp);
  graph.moveIntoGraph(std::move(reduceOpUp));

  logging::pattern::trace("Connecting input {} to {} at {}",
                          weightGradId,
                          reduceOp->str(),
                          ReplicatedAllReduceOp::getInIndex());
  reduceOp->connectInTensor(ReplicatedAllReduceOp::getInIndex(), weightGradId);

  // The reduced gradient
  TensorId reducedGradId =
      !outputId.empty()
          ? outputId
          : graph.getIr().createIntermediateTensorId(weightGradId);

  reduceOp->createAndConnectOutTensor(ReplicatedAllReduceOp::getOutIndex(),
                                      reducedGradId);

  reduceOp->setup();

  graph.topoCons->transfer(combo, reduceOp);
  return reducedGradId;
}

TensorId OptimizerDecompose::gradCast(Graph &graph,
                                      Op *combo,
                                      TensorId gradIntoAcclId,
                                      bool gradAccum) const {
  auto gradType   = DataType::FLOAT;
  auto gradCastUp = std::make_unique<CastOp>(
      Onnx::Operators::Cast_9,
      gradType,
      Op::Settings(
          graph, combo->name() + "_gradCast", combo->settings.debugInfoId));
  auto gradCastOp = gradCastUp.get();
  transferBaseProperties(combo, gradCastOp);
  graph.moveIntoGraph(std::move(gradCastUp));

  gradCastOp->connectInTensor(CastOp::getInIndex(), gradIntoAcclId);

  // The updated gradIntoAcclId
  gradIntoAcclId = graph.getIr().createIntermediateTensorId(gradIntoAcclId);
  gradCastOp->createAndConnectOutTensor(CastOp::getOutIndex(), gradIntoAcclId);

  gradCastOp->setup();

  gradCastOp->settings.optimizerOp = true;

  if (gradAccum) {
    gradCastOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    gradCastOp->setExecutionPhase({});
    gradCastOp->settings.schedulePriority = 0.0;
  }
  return gradIntoAcclId;
}

TensorId OptimizerDecompose::gradUnscale(Graph &graph,
                                         Op *combo,
                                         const OptimizerValue &gs,
                                         TensorId gsId,
                                         TensorId gradIntoAcclId,
                                         bool gradAccum) const {
  Op *gradUnscaleOp;
  OutIndex gradUnscaleOutIdx = 0;

  if (gs.isConst()) {
    auto gradUnscaleUp =
        std::make_unique<ScaleOp>(Onnx::AiGraphcore::OpSet1::Scale,
                                  gs.val(),
                                  Op::Settings(graph,
                                               combo->name() + "_gradUnscale",
                                               combo->settings.debugInfoId));
    gradUnscaleOp = gradUnscaleUp.get();
    transferBaseProperties(combo, gradUnscaleOp);
    graph.moveIntoGraph(std::move(gradUnscaleUp));
    gradUnscaleOp->connectInTensor(ScaleOp::getInIndex(), gradIntoAcclId);
    gradUnscaleOutIdx = ScaleOp::getOutIndex();
  } else {
    auto gradUnscaleUp =
        std::make_unique<MulOp>(Onnx::Operators::Mul_7,
                                Op::Settings(graph,
                                             combo->name() + "_gradUnscale",
                                             combo->settings.debugInfoId));
    gradUnscaleOp = gradUnscaleUp.get();
    transferBaseProperties(combo, gradUnscaleOp);
    graph.moveIntoGraph(std::move(gradUnscaleUp));

    gradUnscaleOp->connectInTensor(MulOp::getArg0InIndex(), gradIntoAcclId);
    gradUnscaleOp->connectInTensor(MulOp::getArg1InIndex(), gsId);
    gradUnscaleOutIdx = MulOp::getOutIndex();
  }

  // The updated gradIntoAcclId
  gradIntoAcclId = graph.getIr().createIntermediateTensorId(gradIntoAcclId);
  gradUnscaleOp->createAndConnectOutTensor(gradUnscaleOutIdx, gradIntoAcclId);

  gradUnscaleOp->setup();
  gradUnscaleOp->settings.optimizerOp = true;

  if (gradAccum) {
    gradUnscaleOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    gradUnscaleOp->setExecutionPhase({});
    gradUnscaleOp->settings.schedulePriority = 0.0;
  }
  return gradIntoAcclId;
}

TensorId OptimizerDecompose::regularizeL2(Graph &graph,
                                          Op *combo,
                                          const OptimizerValue &wd,
                                          TensorId wdId,
                                          TensorId weightId,
                                          TensorId gradIntoAcclId,
                                          bool gradAccum) const {
  Op *scaledAddOp;

  if (wd.isConst()) {
    auto scaledAddUp = std::make_unique<ScaledAddOp>(
        Onnx::AiGraphcore::OpSet1::ScaledAdd,
        1.0f,
        wd.val(),
        Op::Settings(graph,
                     combo->name() + "_weightDecayScale",
                     combo->settings.debugInfoId));
    scaledAddOp = scaledAddUp.get();
    transferBaseProperties(combo, scaledAddOp);
    graph.moveIntoGraph(std::move(scaledAddUp));
  } else {
    auto scaledAddUp = std::make_unique<ScaledAddOp>(
        Onnx::AiGraphcore::OpSet1::ScaledAdd,
        1.0f,
        1.0f,
        Op::Settings(graph,
                     combo->name() + "_weightDecayScale",
                     combo->settings.debugInfoId));
    scaledAddOp = scaledAddUp.get();
    transferBaseProperties(combo, scaledAddOp);
    graph.moveIntoGraph(std::move(scaledAddUp));
    scaledAddOp->connectInTensor(ScaledAddOp::getScale1InIndex(), wdId);
  }

  scaledAddOp->connectInTensor(ScaledAddOp::getArg0InIndex(), gradIntoAcclId);
  scaledAddOp->connectInTensor(ScaledAddOp::getArg1InIndex(), weightId);

  // The updated gradIntoAcclId
  gradIntoAcclId = graph.getIr().createIntermediateTensorId(gradIntoAcclId);
  scaledAddOp->createAndConnectOutTensor(ScaledAddOp::getOutIndex(),
                                         gradIntoAcclId);

  scaledAddOp->setup();
  scaledAddOp->settings.optimizerOp = true;

  if (gradAccum) {
    scaledAddOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    scaledAddOp->setExecutionPhase({});
    scaledAddOp->settings.schedulePriority = 0.0;
  }
  return gradIntoAcclId;
}

} // namespace popart
