// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/accumulatorupdate.hpp>
#include <popart/op/add.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/convbase.hpp>
#include <popart/op/div.hpp>
#include <popart/op/histogram.hpp>
#include <popart/op/lossscaleupdate.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/mul.hpp>
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

/**
 * Here we determine the levels, or histogram bin edges for the Histogram ops
 * inserted by the transform. Design decision (2): We choose a single bin edge,
 * whose value is some factor of the maximum value that can be represented by
 * the tensor's data type.
 *
 * \param tensor A pointer to the tensor for which a Histogram op is created.
 *     Its dtype is used to determine numeric_limits.max().
 * \param binEdgeLocation A factor in [0, 1] that moves the bin edge between
 *     [0, numeric_limits.max()].
 * \return A vector of floats with the histogram bin edges.
 */
std::vector<float> getLevels(Tensor *tensor, float binEdgeLocation) {
  if (binEdgeLocation < 0 || binEdgeLocation > 1) {
    throw error(
        "[AutomaticLossScale transform] Out of range value for "
        "'binEdgeLocation'. The current value is {}, but it should be in "
        "the range [0, 1].",
        binEdgeLocation);
  }

  auto dtype = tensor->info.dataType();
  if (dtype == DataType::FLOAT) {
    return {std::numeric_limits<float>::max() * binEdgeLocation};
  } else if (dtype == DataType::FLOAT16) {
    return {static_cast<float>(std::numeric_limits<uint16_t>::max()) *
            binEdgeLocation};
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

OptionalPipelineStage findLowestPipelineStage(std::vector<Tensor *> tensors) {
  std::set<PipelineStage> allStages;
  for (Tensor *tensor : tensors) {
    std::set<PipelineStage> stages = tensor->consumers.getPipelineStages();
    for (auto stage : stages) {
      allStages.insert(stage);
    }
  }

  if (allStages.size() == 0) {
    return {};
  } else {
    return *std::min_element(allStages.begin(), allStages.end());
  }
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
  return reservedAccumPrefix() + std::string("autoLossScaleStats");
}

TensorId getLossScaleUpdateFactorTensorId() {
  return std::string("lossScaleUpdateFactor");
}

Tensor *
addOnesTensor(Graph &graph, const TensorId &tensorId, const TensorInfo info) {
  auto &ir = graph.getIr();
  if (ir.tensorExistsInInitialisers(tensorId)) {
    auto tp = onnxutil::getTensorProto(ir.getModel(), tensorId);
    graph.getTensors().addVarInit(tensorId, &tp);
  } else {
    if (info.dataType() == DataType::FLOAT) {
      std::vector<float> d(info.nelms(), 1.0f);
      graph.getTensors().addVarInit(tensorId, info, d.data());
    } else if (info.dataType() == DataType::FLOAT16) {
      std::vector<float16_t> d(info.nelms(), static_cast<float16_t>(1.0));
      graph.getTensors().addVarInit(tensorId, info, d.data());
    } else {
      throw error("[AutomaticLossScale transform] Can only create a 'ones' "
                  "tensor of FLOAT16 and FLOAT data types.");
    }
  }

  return graph.getTensors().get(tensorId);
}

std::set<Tensor *>
AutomaticLossScale::getInverseLossScaleTensors(const Graph &graph) {
  const Ir &ir               = graph.getIr();
  const Optimizer &optimizer = ir.getOptimizer();

  // To ensure that the tensor we return from this method is the compound
  // scalar this is used to apply the inverse loss scale in all VarUpdateOps
  // in this graph, we check that all Variable tensors have the same type.
  // Otherwise the graph will contain more than one of these tensors; one
  // per type.
  auto variables = graph.getTensors().getOfType(TensorType::Variable);

  std::set<Tensor *> inverseLossScaleTensors;
  for (Tensor *variable : variables) {
    if (ir.tensorExistsInInitialisers(variable->id)) {
      TensorId inverseLossScaleId =
          optimizer.getInverseLossScalingTensorId(*variable);
      if (graph.getTensors().contains(inverseLossScaleId)) {
        inverseLossScaleTensors.insert(
            graph.getTensors().get(inverseLossScaleId));
      } else {
        throw error("[AutomaticLossScale transform] Unable to find inverse "
                    "loss scale tensor, '{}', in graph '{}'",
                    inverseLossScaleId,
                    graph.id);
      }
    }
  }

  return inverseLossScaleTensors;
}

bool AutomaticLossScale::apply(Graph &graph) const {
  // Some checks:
  // 1. Must be a training session
  // 2. The optimizer's loss scaling is non-const
  // 3. Not compatible with continuous pipelining

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
  if (ir.getSessionOptions().enablePipelining == true &&
      !ir.getSessionOptions().enableGradientAccumulation) {
    throw error("[AutomaticLossScale transform] Automatic loss scaling is not "
                "currently supported when the 'enablePipelining' SessionOption "
                "is set to 'true', but the 'enableGradientAccumulation' "
                "SessionOption is set to 'false'");
  }

  // Get all tensors whose statistics are to be tracked.
  std::vector<Tensor *> toTrackTensors = getToTrackTensors(graph);
  std::vector<Tensor *> histogramOutputs;
  for (Tensor *tensor : toTrackTensors) {
    logging::transform::debug("Collecting statistics for tensor '{}' for "
                              "control of loss-scale value.",
                              tensor->id);

    // Get automatic loss scaling hyperparameters.
    float binEdgeLocation =
        ir.getSessionOptions().automaticLossScalingSettings.binEdgeLocation;

    // Attach a newly created HistogramOp to each tensor
    auto histogramOp =
        graph.createOp<HistogramOp>(Onnx::CustomOperators::Histogram,
                                    getLevels(tensor, binEdgeLocation),
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
  // the tensors to be updated
  Tensor *lossScaleTensor = getLossScaleTensor(graph);
  std::set<Tensor *> inverseLossScaleTensors =
      getInverseLossScaleTensors(graph);

  // Pass loss scale tensor and HistogramOp outputs into the LossScaleUpdateOp
  auto lossScaleUpdateOp =
      graph.createOp<LossScaleUpdateOp>(Onnx::CustomOperators::LossScaleUpdate,
                                        lossScaleTensor->info.dataType(),
                                        Op::Settings(graph, ""));

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

  // Create variable tensor, initialised to 'ones' for input at
  // LossScaleUpdateOp::getLossScaleUpdateFactorInIndex
  Tensor *lsUpdateFactor = addOnesTensor(
      graph, getLossScaleUpdateFactorTensorId(), lossScaleTensor->info);

  lossScaleUpdateOp->connectInTensor(
      LossScaleUpdateOp::getLossScaleUpdateFactorInIndex(), lsUpdateFactor->id);
  lossScaleUpdateOp->createAndConnectOutTensor(
      LossScaleUpdateOp::getUpdatedLossScaleUpdateFactorOutIndex(),
      reservedUpdatedVarPrefix() + lsUpdateFactor->id);
  lossScaleUpdateOp->setup();
  lossScaleUpdateOp->pruneable = false;

  // Case 2 or 3, execute the loss scale update in the outer fragment
  if (doingGradientAccumulation(graph)) {
    OptionalPipelineStage optionalPs;
    lossScaleUpdateOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
    lossScaleUpdateOp->setPipelineStage(optionalPs);
  }
  // Case 0 or 1
  else {
    lossScaleUpdateOp->settings.executionContext = ExecutionContext::Normal;
  }

  std::vector<Op *> preLossScaleUpdateOps;
  // Apply the loss scale update factor to:
  // - loss scale tensor
  // - the inverse loss scale tensor(s)

  auto lsMulOp =
      graph.createOp<MulOp>(Onnx::AiOnnx::OpSet6::Mul, Op::Settings(graph, ""));
  lsMulOp->connectInTensor(MulOp::getArg0InIndex(), lossScaleTensor->id);
  lsMulOp->connectInTensor(MulOp::getArg1InIndex(), lsUpdateFactor->id);
  // Design note:
  // Output name cannot contain any of reservedOptimizerPrefixes(),
  // otherwise it is treated as a tensor that is written to as part of
  // the OptimizerFromHost fragment.
  lsMulOp->createAndConnectOutTensor(MulOp::getOutIndex(), "finalLossScale");
  lsMulOp->setup();
  preLossScaleUpdateOps.push_back(lsMulOp);
  auto lossScaleUpdated = lsMulOp->outTensor(MulOp::getOutIndex());

  // Optionally set VirtualGraphId, PipelineStage
  auto vgId   = lossScaleTensor->consumers.findLowestVirtualGraphID();
  auto pStage = lossScaleTensor->consumers.findLowestPipelineStage();

  // Disconnect the loss scale tensor from its consumers and reconnect to the
  // updated loss scale tensor
  for (Op *consumer : lossScaleTensor->consumers.getOps()) {
    if (consumer != lsMulOp) {
      auto inIndices = consumer->input->indices(lossScaleTensor);
      for (auto inIndex : inIndices) {
        consumer->disconnectInTensor(inIndex);
        consumer->connectInTensor(inIndex, lossScaleUpdated->id);
      }
    }
  }

  // Sort inverse loss scale tensors by data type
  std::map<DataType, std::vector<Tensor *>> inverseLossScaleTensorsMap;
  for (Tensor *tensor : inverseLossScaleTensors) {
    const auto iter = inverseLossScaleTensorsMap.find(tensor->info.dataType());
    if (iter == inverseLossScaleTensorsMap.end()) {
      inverseLossScaleTensorsMap.emplace(tensor->info.dataType(),
                                         std::vector<Tensor *>{tensor});
    } else {
      iter->second.push_back(tensor);
    }
  }

  for (auto dtype_tensors : inverseLossScaleTensorsMap) {
    DataType dtype                = dtype_tensors.first;
    std::vector<Tensor *> tensors = dtype_tensors.second;

    Tensor *finalLossScaleUpdateFactor = nullptr;
    if (dtype != lsUpdateFactor->info.dataType()) {
      auto castOp = graph.createOp<CastOp>(
          Onnx::Operators::Cast_9, dtype, Op::Settings(graph, ""));
      castOp->connectInTensor(CastOp::getInIndex(), lsUpdateFactor->id);
      castOp->createAndConnectOutTensor(CastOp::getOutIndex(),
                                        lsUpdateFactor->id + "_cast");
      castOp->setup();
      preLossScaleUpdateOps.push_back(castOp);
      finalLossScaleUpdateFactor = castOp->outTensor(CastOp::getOutIndex());

      // Optionally set VirtualGraphId, PipelineStage
      castOp->setVirtualGraphId(lossScaleUpdateOp->getOptionalVGraphId());
      castOp->setPipelineStage(findLowestPipelineStage(tensors));

    } else {
      finalLossScaleUpdateFactor = lsUpdateFactor;
    }

    for (int i = 0; i < tensors.size(); i++) {
      Tensor *inverseLossScaleTensor = tensors.at(i);
      auto lsDivOp = graph.createOp<DivOp>(Onnx::AiOnnx::OpSet6::Div,
                                           Op::Settings(graph, ""));
      lsDivOp->connectInTensor(DivOp::getArg0InIndex(),
                               inverseLossScaleTensor->id);
      lsDivOp->connectInTensor(DivOp::getArg1InIndex(),
                               finalLossScaleUpdateFactor->id);
      // Design note:
      // Output name cannot contain any of reservedOptimizerPrefixes(),
      // otherwise it is treated as a tensor that is written to as part of
      // the OptimizerFromHost fragment.
      lsDivOp->createAndConnectOutTensor(
          DivOp::getOutIndex(), "finalInverseLossScale_" + std::to_string(i));
      lsDivOp->setup();
      preLossScaleUpdateOps.push_back(lsDivOp);
      auto inverseLossScaleUpdated = lsDivOp->outTensor(DivOp::getOutIndex());

      // Optionally set VirtualGraphId, PipelineStage
      lsDivOp->setVirtualGraphId(
          inverseLossScaleTensor->consumers.findLowestVirtualGraphID());
      lsDivOp->setPipelineStage(
          inverseLossScaleTensor->consumers.findLowestPipelineStage());

      // Disconnect the inverse loss scale tensor from its consumers and
      // reconnect to the updated inverse loss scale tensor
      for (Op *consumer : inverseLossScaleTensor->consumers.getOps()) {
        if (consumer != lsDivOp) {
          auto inIndices = consumer->input->indices(inverseLossScaleTensor);
          for (auto inIndex : inIndices) {
            consumer->disconnectInTensor(inIndex);
            consumer->connectInTensor(inIndex, inverseLossScaleUpdated->id);
          }
        }
      }
    }
  }

  // Case 0 or 1
  if (!doingGradientAccumulation(graph)) {
    for (Op *op : preLossScaleUpdateOps) {
      graph.topoCons->insert(op, lossScaleUpdateOp);
    }
  }

  // Optionally set VirtualGraphId, PipelineStage
  for (Op *op : preLossScaleUpdateOps) {
    op->setVirtualGraphId(vgId);
    op->setPipelineStage(pStage);
  }
  lossScaleUpdateOp->setVirtualGraphId(vgId);

  return true;
}

namespace {
bool init = Transform::registerTransform(new AutomaticLossScale);
}

} // namespace popart
