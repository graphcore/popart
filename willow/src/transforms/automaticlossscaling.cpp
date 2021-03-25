// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulatorupdate.hpp>
#include <popart/op/add.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/convbase.hpp>
#include <popart/op/histogram.hpp>
#include <popart/op/lossscaleupdate.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/sum.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensorindex.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/automaticlossscaling.hpp>

namespace popart {

namespace {

bool producesToTrackTensors(Op *op) {
  // conv or matmul operations that produce gradient tensors
  if (op->fromLoss == PathFromLoss::Yes) {
    if (op->isConvertibleTo<MultiConvBaseOp>()) {
      return true;
    } else if (op->isConvertibleTo<MultiConvWeightsGradBaseOp>()) {
      return true;
    } else if (op->isConvertibleTo<MatMulOp>()) {
      return true;
    }
  }

  // Check for error cases. A sanity check. If we find the below ops, then
  // something has gone wrong outside of this transform.
  if (op->isConvertibleTo<MatMulLhsGradOp>() ||
      op->isConvertibleTo<MatMulRhsGradOp>() ||
      op->isConvertibleTo<MultiConvDataGradBaseOp>()) {
    throw internal_error(
        "[AutomaticLossScale transform] Unexpected Op '{}' found when looking "
        "through producers of tracked tensors");
  }

  return false;
}

std::vector<Tensor *> getToTrackTensors(Graph &graph) {
  std::vector<Tensor *> toTrackTensors;

  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();

    if (producesToTrackTensors(op)) {
      for (Tensor *tensor : op->output->tensors()) {
        toTrackTensors.push_back(tensor);
      }
    }
  }

  // Verify that we are returning a non-empty vector of to-track tensors
  if (toTrackTensors.size() == 0) {
    throw error("[AutomaticLossScale transform] No tracked tensors were found");
  }

  return toTrackTensors;
}

// A tensor has (or is close to having) overflow if some of its elements have
// a very large +ve or very large -ve value. Design decision (1): we take the
// absolute value of a tensor sorting its elements into bins, so we only have
// to look for the number of very large +ve elements.
bool absoluteOfInput() { return true; }

// Here we hard-code the levels, or histogram bin edges for the Histogram ops
// inserted by the transform. Design decision (2): We choose a single bin edge,
// whose value is some factor of the maximum value that can be represented by
// the tensor's data type.
std::vector<float> getLevels(Tensor *tensor) {
  auto dtype = tensor->info.dataType();
  if (dtype == DataType::FLOAT) {
    return {std::numeric_limits<float>::max() / 2};
  } else if (dtype == DataType::FLOAT16) {
    return {static_cast<float>(std::numeric_limits<uint16_t>::max()) / 2};
  } else {
    throw error("[AutomaticLossScale transform] Unsupported data type {} for "
                "to-track tensor '{}'",
                dtype,
                tensor->id);
  }
}

bool doingGraphReplication(const Graph &graph) {
  auto &ir = graph.getIr();

  return ir.getSessionOptions().enableReplicatedGraphs &&
         ir.getSessionOptions().replicatedGraphCount > 1;
}

bool doingGradientAccumulation(const Graph &graph) {
  auto &ir = graph.getIr();

  return ir.getSessionOptions().enableGradientAccumulation &&
         ir.getSessionOptions().accumulationFactor > 1;
}

} // namespace

std::size_t AutomaticLossScale::id() {
  return typeid(AutomaticLossScale).hash_code();
}

Tensor *AutomaticLossScale::getLossScaleTensor(const Graph &graph) {
  const Ir &ir               = graph.getIr();
  const Optimizer &optimizer = ir.getOptimizer();

  TensorId lsFP16 = optimizer.getLossScalingTensorId(DataType::FLOAT16);
  TensorId lsFP32 = optimizer.getLossScalingTensorId(DataType::FLOAT);
  bool existsLossScaleFP16 = graph.getTensors().contains(lsFP16);
  bool existsLossScaleFP32 = graph.getTensors().contains(lsFP32);

  Tensor *lossScaleTensor;
  if (existsLossScaleFP16 && existsLossScaleFP32) {
    throw error("[AutomaticLossScale transform] Unable to determine the data "
                "type of the loss scale tensor, as both tensors '{}' and '{}' "
                "exist in graph {}",
                lsFP16,
                lsFP32,
                graph.id);
  } else {
    if (existsLossScaleFP16) {
      lossScaleTensor = graph.getTensors().get(lsFP16);
    } else if (existsLossScaleFP32) {
      lossScaleTensor = graph.getTensors().get(lsFP32);
    } else {
      throw error("[AutomaticLossScale transform] Unable to find any loss "
                  "scale tensor in graph '{}'",
                  graph.id);
    }
  }

  return lossScaleTensor;
}

TensorId getStatisticsAccumulatorTensorId() {
  return reservedAcclPrefix() + std::string("autoLossScaleStats");
}

Tensor *AutomaticLossScale::getInverseLossScaleTensor(const Graph &graph) {
  const Ir &ir               = graph.getIr();
  const Optimizer &optimizer = ir.getOptimizer();

  if (optimizer.type() == OptimizerType::SGD) {
    auto sgd = static_cast<const SGD &>(optimizer);

    // We assume that all VarUpdateOps consume a single compound scalar tensor
    // that contains the inverse scale factor. So none of the atomic scalars
    // that are combined to produce this compound scalar can have a per-weight
    // specific value. Therefore we perform the (stricter than necessary) check
    // that no specific optimizer values have been added.
    if (sgd.hasSpecific()) {
      throw error("[AutomaticLossScale transform] Not compatible with "
                  "weight-specific optimizer values");
    }

    // To ensure that the tensor we return from this method is the compound
    // scalar this is used to apply the inverse loss scale in all VarUpdateOps
    // in this graph, we check that all Variable tensors have the same type.
    // Otherwise the graph will contain more than one of these tensors; one
    // per type.
    auto variables = graph.getTensors().getOfType(TensorType::Variable);
    auto dtype     = variables.front()->info.dataType();
    for (Tensor *variable : variables) {
      if (variable->id == getStatisticsAccumulatorTensorId()) {
        continue;
      }

      if (variable->info.dataType() != dtype) {
        throw error("[AutomaticLossScale transform] All Variable tensors must "
                    "have the same data type to ensure there is only one "
                    "inverse loss scale tensor used in the graph. Tensor '{}' "
                    "has dtype {}, but tensor '{}' has dtype {}",
                    variable->id,
                    variable->info.data_type(),
                    variables.front()->id,
                    variables.front()->info.data_type());
      }
    }

    TensorId inverseLossScaleId =
        sgd.getInverseLossScalingTensorId(*variables.front());

    if (graph.getTensors().contains(inverseLossScaleId)) {
      return graph.getTensors().get(inverseLossScaleId);
    } else {
      throw error("[AutomaticLossScale transform] Unable to find inverse loss "
                  "scale tensor, '{}', in graph '{}'",
                  inverseLossScaleId,
                  graph.id);
    }
  } else {
    throw internal_error("[AutomaticLossScale transform] Only SGD supported");
  }
}

bool AutomaticLossScale::apply(Graph &graph) const {
  // Some checks:
  // 1. Must be a training session
  // 2. The optimizer's loss scaling is non-const
  // 3. Not compatible with pipelining: (T33956)
  // 4. Not compatible with non-SGD optimizer

  // 1.
  auto &ir = graph.getIr();
  if (!ir.canTrain()) {
    throw error(
        "[AutomaticLossScale transform] Only compatible when doing training");
  }

  // 2.
  if (ir.getOptimizer().lossScaling().isConst()) {
    throw error("[AutomaticLossScale transform] The optimizer must have "
                "non-const loss scaling");
  }

  // 3.
  if (ir.getSessionOptions().enablePipelining == true) {
    throw error("[AutomaticLossScale transform] Automatic loss scaling is not "
                "currently supported when the 'enablePipelining' SessionOption "
                "is set to 'true'");
  }

  // 4.
  if (ir.getOptimizer().type() != OptimizerType::SGD) {
    throw error("[AutomaticLossScale transform] Only compatible when using the "
                "SGD optimizer type, but you are using '{}'",
                ir.getOptimizer().type_s());
  }

  // Get all tensors whose statistics are to be tracked.
  std::vector<Tensor *> toTrackTensors = getToTrackTensors(graph);
  std::vector<Tensor *> histogramOutputs;
  for (Tensor *tensor : toTrackTensors) {
    logging::transform::debug("Collecting statistics for tensor '{}' for "
                              "control of loss-scale value.",
                              tensor->id);

    // Attach a newly created HistogramOp to each tensor
    auto histogramOp =
        graph.createOp<HistogramOp>(Onnx::CustomOperators::Histogram,
                                    getLevels(tensor),
                                    absoluteOfInput(),
                                    Op::Settings(graph, ""));

    histogramOp->connectInTensor(HistogramOp::getInIndex(), tensor->id);
    histogramOp->createAndConnectOutTensor(HistogramOp::getOutIndex(),
                                           tensor->id + "_statistics");
    histogramOp->inheritPlacementAttributes(false);
    histogramOp->setup();
    histogramOutputs.push_back(
        histogramOp->outTensor(HistogramOp::getOutIndex()));
  }

  // Get the loss scale tensor and the inverse loss scale tensor:
  // the tensors to updated
  Tensor *lossScaleTensor         = getLossScaleTensor(graph);
  Tensor *inverseLossScaleTensor  = getInverseLossScaleTensor(graph);
  auto originalLossScaleConsumers = lossScaleTensor->consumers.getOps();
  for (Op *op : inverseLossScaleTensor->consumers.getOps()) {
    originalLossScaleConsumers.push_back(op);
  }

  // Pass loss scale tensor and HistogramOp outputs into the LossScaleUpdateOp
  auto lossScaleUpdateOp = graph.createOp<LossScaleUpdateOp>(
      Onnx::CustomOperators::LossScaleUpdate, Op::Settings(graph, ""));

  lossScaleUpdateOp->connectInTensor(LossScaleUpdateOp::getLossScaleInIndex(),
                                     lossScaleTensor->id);
  lossScaleUpdateOp->connectInTensor(
      LossScaleUpdateOp::getInverseLossScaleInIndex(),
      inverseLossScaleTensor->id);

  // The grad statistics tensors to connect to the LossScaleUpdateOp
  std::vector<Tensor *> finalStatisticsTensors;

  // Case 0
  if (!(doingGraphReplication(graph) || doingGradientAccumulation(graph))) {
    finalStatisticsTensors = histogramOutputs;
  }

  // Case 1, 2 or 3: Sum the statistics tensors
  else {
    // Sum the histogram tensors
    auto statsSumOp =
        graph.createOp<SumOp>(Onnx::Operators::Sum_8, Op::Settings(graph, ""));

    for (int i = 0; i < histogramOutputs.size(); i++) {
      Tensor *tensor = histogramOutputs.at(i);
      statsSumOp->connectInTensor(i, tensor->id);
    }

    statsSumOp->createAndConnectOutTensor(SumOp::getOutIndex(),
                                          "summedHistograms");
    statsSumOp->inheritPlacementAttributes(false);
    statsSumOp->setup();

    finalStatisticsTensors = {statsSumOp->outTensor(SumOp::getOutIndex())};
  }

  // Case 2 or 3, Accumulate the summed gradient statistics in the inner loop,
  // reset the accumulator in the outer loop.
  if (doingGradientAccumulation(graph)) {
    Tensor *inTensor = finalStatisticsTensors.back();

    // 1. add variable accl tensor, set tensor data to zeros
    TensorId toAcclId = getStatisticsAccumulatorTensorId();
    std::vector<uint32_t> d(inTensor->info.nelms(), 0);
    graph.getTensors().addVarInit(toAcclId, inTensor->info, d.data());

    // 2. add op to accumulate
    auto statsAcclOp = graph.createOp<AddRhsInplaceOp>(Op::Settings(graph, ""));
    statsAcclOp->connectInTensor(AddRhsInplaceOp::getArg0InIndex(),
                                 inTensor->id);
    statsAcclOp->connectInTensor(AddRhsInplaceOp::getArg1InIndex(), toAcclId);
    statsAcclOp->createAndConnectOutTensor(AddRhsInplaceOp::getOutIndex(),
                                           inTensor->id + "_accld");
    statsAcclOp->inheritPlacementAttributes(false);
    statsAcclOp->setup();
    TensorId accldId = statsAcclOp->outId(AddRhsInplaceOp::getOutIndex());

    // 3. add AccumulatorUpdateOp to reset the accl tensor
    auto statsAcclResetOp = graph.createOp<AccumulatorUpdateOp>(
        OptimizerValue(0.0f), Op::Settings(graph, ""));
    statsAcclResetOp->connectInTensor(
        AccumulatorUpdateOp::getVarToUpdateInIndex(), toAcclId);
    statsAcclResetOp->createAndConnectOutTensor(
        AccumulatorUpdateOp::getUpdatedVarOutIndex(),
        inTensor->id + "_accld_reset");
    statsAcclResetOp->inheritPlacementAttributes(false);
    statsAcclResetOp->setup();

    // Reset the accumulator, and update the loss in the outer fragment
    statsAcclResetOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    lossScaleUpdateOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;

    graph.topoCons->insert(lossScaleUpdateOp, statsAcclResetOp);

    finalStatisticsTensors = {
        statsAcclOp->outTensor(AddRhsInplaceOp::getOutIndex())};
  }

  // Case 1 or 3, Reduce the summed or accumulated statistics tensor
  if (doingGraphReplication(graph)) {
    TensorId inId = finalStatisticsTensors.back()->id;

    auto reduceOp = graph.createOp<ReplicatedAllReduceOp>(
        Onnx::CustomOperators::ReplicatedAllReduce,
        Op::Settings(graph, inId + "_reduced"));

    reduceOp->connectInTensor(ReplicatedAllReduceOp::getInIndex(), inId);
    reduceOp->createAndConnectOutTensor(ReplicatedAllReduceOp::getOutIndex(),
                                        "summedHistograms_reduced");
    reduceOp->inheritPlacementAttributes(false);
    reduceOp->setup();

    // Case 3
    if (doingGradientAccumulation(graph)) {
      reduceOp->settings.executionContext =
          ExecutionContext::AccumulateOuterFragment;
    }
    // Case 1
    else {
      reduceOp->settings.executionContext = ExecutionContext::Normal;
    }

    finalStatisticsTensors = {
        reduceOp->outTensor(ReplicatedAllReduceOp::getOutIndex())};
  }

  for (int i = 0; i < finalStatisticsTensors.size(); i++) {
    Tensor *tensor = finalStatisticsTensors.at(i);
    lossScaleUpdateOp->connectInTensor(
        i + LossScaleUpdateOp::getFirstStatisticsTensorInIndex(), tensor->id);
  }

  if (lossScaleTensor->hasVirtualGraphId()) {
    lossScaleUpdateOp->setVirtualGraphId(lossScaleTensor->getVirtualGraphId());
  }

  lossScaleUpdateOp->createAndConnectOutTensor(
      LossScaleUpdateOp::getUpdatedLossScaleOutIndex(),
      lossScaleTensor->id + "_updated");
  lossScaleUpdateOp->createAndConnectOutTensor(
      LossScaleUpdateOp::getUpdatedInverseLossScaleOutIndex(),
      inverseLossScaleTensor->id + "_updated");
  lossScaleUpdateOp->setup();

  // Case 2 or 3, execute the loss scale update in the outer fragment
  if (doingGradientAccumulation(graph)) {
    lossScaleUpdateOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  }
  // Case 0 or 1
  else {
    lossScaleUpdateOp->settings.executionContext = ExecutionContext::Normal;
  }

  // Ensure that all other consumers of loss scale tensor run before the loss
  // scale update runs. The loss is scaled at the start of the backwards pass,
  // and then inversely right before the they are applied as a weight update.
  // The loss scale tensor must have the same value at these two points.
  for (Op *consumer : originalLossScaleConsumers) {
    graph.topoCons->insert(consumer, lossScaleUpdateOp);
  }
  lossScaleUpdateOp->pruneable = false;

  return true;
}

namespace {
bool init = Transform::registerTransform(new AutomaticLossScale);
}

} // namespace popart
