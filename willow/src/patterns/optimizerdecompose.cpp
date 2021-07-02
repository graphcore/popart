// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <onnxutil.hpp>
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

namespace popart {

template <typename T>
void OptimizerDecompose::addStateTensor(Graph &graph,
                                        const TensorId &tensorId,
                                        const TensorInfo info,
                                        float initValue) const {
  auto &ir = graph.getIr();
  if (ir.tensorExistsInInitialisers(tensorId)) {
    auto tp = onnxutil::getTensorProto(ir.getModel(), tensorId);
    graph.getTensors().addVarInit(tensorId, &tp);
  } else {
    std::vector<T> d(info.nelms(), static_cast<T>(initValue));
    graph.getTensors().addVarInit(tensorId, info, d.data());
  }
}

template void
OptimizerDecompose::addStateTensor<float>(Graph &graph,
                                          const TensorId &tensorId,
                                          const TensorInfo info,
                                          float initValue) const;

template void
OptimizerDecompose::addStateTensor<float16_t>(Graph &graph,
                                              const TensorId &tensorId,
                                              const TensorInfo info,
                                              float initValue) const;

TensorInfo OptimizerDecompose::addStateTensor(Graph &graph,
                                              const TensorId &tensorId,
                                              const Shape &shape,
                                              const DataType &type,
                                              float initValue) const {
  auto info = TensorInfo(type, shape);
  switch (type) {
  case DataType::FLOAT:
    addStateTensor<float>(graph, tensorId, info, initValue);
    break;
  case DataType::FLOAT16:
    addStateTensor<float16_t>(graph, tensorId, info, initValue);
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
      type, value, Op::Settings(graph, combo->name() + acclName));
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

TensorId OptimizerDecompose::gradAccum(Graph &graph,
                                       Op *combo,
                                       TensorId accumId,
                                       TensorId gradIntoAccumId,
                                       bool accumReduce,
                                       TensorId outputId) const {
  TensorId gradIntoAcclId;
  auto accumOpUp = std::make_unique<AccumulateOp>(
      AccumulationType::Add,
      OptimizerValue(1.0f),
      Op::Settings(graph, combo->name() + "_accumulate"));
  auto accumOp = accumOpUp.get();
  transferBaseProperties(combo, accumOp);
  graph.moveIntoGraph(std::move(accumOpUp));

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
    auto reduceOpUp = std::make_unique<ReplicatedAllReduceInplaceOp>(
        Onnx::CustomOperators::ReplicatedAllReduceInplace,
        Op::Settings(graph, combo->name() + "_reduce"));
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
  auto accumZeroOpUp = std::make_unique<AccumulatorZeroOp>(
      Op::Settings(graph, combo->name() + "_accumupdate"));
  auto accumZeroOp = accumZeroOpUp.get();
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
                                        TensorId weightGradId,
                                        TensorId outputId) const {
  auto reduceOpUp = std::make_unique<ReplicatedAllReduceOp>(
      Onnx::CustomOperators::ReplicatedAllReduce,
      Op::Settings(graph, combo->name() + "_reduce"));
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
      Op::Settings(graph, combo->name() + "_gradCast"));
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
    auto gradUnscaleUp = std::make_unique<ScaleOp>(
        Onnx::AiGraphcore::OpSet1::Scale,
        gs.val(),
        Op::Settings(graph, combo->name() + "_gradUnscale"));
    gradUnscaleOp = gradUnscaleUp.get();
    transferBaseProperties(combo, gradUnscaleOp);
    graph.moveIntoGraph(std::move(gradUnscaleUp));
    gradUnscaleOp->connectInTensor(ScaleOp::getInIndex(), gradIntoAcclId);
    gradUnscaleOutIdx = ScaleOp::getOutIndex();
  } else {
    auto gradType = graph.getTensors().get(gradIntoAcclId)->info.dataType();
    auto gsType   = graph.getTensors().get(gsId)->info.dataType();
    if (gradType != gsType) {
      auto gsCastId = graph.getIr().createIntermediateTensorId("gsCast");
      auto gsCast   = graph.createConnectedOp<CastOp>(
          {{CastOp::getInIndex(), gsId}},
          {{CastOp::getOutIndex(), gsCastId}},
          Onnx::Operators::Cast_9,
          gradType,
          Op::Settings(graph, combo->name() + "_gsCast"));
      transferBaseProperties(combo, gsCast);
      gsId = gsCastId;

      if (gradAccum) {
        gsCast->settings.executionContext =
            ExecutionContext::AccumulateOuterFragment;
        gsCast->setExecutionPhase({});
        gsCast->settings.schedulePriority = 0.0;
      }
    }

    auto gradUnscaleUp = std::make_unique<MulOp>(
        Onnx::Operators::Mul_7,
        Op::Settings(graph, combo->name() + "_gradUnscale"));
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
        Op::Settings(graph, combo->name() + "_weightDecayScale"));
    scaledAddOp = scaledAddUp.get();
    transferBaseProperties(combo, scaledAddOp);
    graph.moveIntoGraph(std::move(scaledAddUp));
  } else {
    auto scaledAddUp = std::make_unique<ScaledAddOp>(
        Onnx::AiGraphcore::OpSet1::ScaledAdd,
        1.0f,
        1.0f,
        Op::Settings(graph, combo->name() + "_weightDecayScale"));
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
