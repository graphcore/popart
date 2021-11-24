// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <onnxutil.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/aliasesmap.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulatorzero.hpp>
#include <popart/op/add.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/convbase.hpp>
#include <popart/op/div.hpp>
#include <popart/op/histogram.hpp>
#include <popart/op/incrementmod.hpp>
#include <popart/op/less.hpp>
#include <popart/op/lossscaleupdate.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/sum.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/automaticlossscaling.hpp>
#include <popart/util.hpp>

#include <popart/transforms/subgraphoutline.hpp>

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

void checkOutputConsumers(Op *op) {
  for (auto &index_tensor : op->output->tensorMap()) {
    auto output = index_tensor.second;
    if (output->consumers.getOps().empty()) {
      throw error(
          "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
          "Output tensor {} of operator {} does not have consumer. "
          "executeOpNTimesEveryMTimes requires that otputs of the operator"
          ", on which we apply executeOpNTimesEveryMTimes, have consumers. ",
          output->id,
          op->str());
    }
  }
}

void checkFrequencyModifierSettings(
    const std::map<InIndex, OutIndex> &identityInputToOutputIndiciesMapping,
    const std::map<OutIndex, float> &outputIndiciesAndValues,
    const std::string &opid) {

  std::set<OutIndex> seenIdentityOutIndices;
  for (const auto &el : identityInputToOutputIndiciesMapping) {
    seenIdentityOutIndices.insert(el.second);
  }

  for (const auto &el : outputIndiciesAndValues) {
    if (seenIdentityOutIndices.count(el.first)) {
      throw error(
          "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
          "Incorrect frequency modifier settings for {} operator "
          "Identity output and default output can not have the same index {}.",
          opid,
          el.first);
    }
  }
}

void checkIdentityInputToOutputIndiciesMapping(
    const std::map<InIndex, OutIndex> &identityInputToOutputIndiciesMapping,
    Op *op) {
  std::set<int> inputIndices;
  for (const auto &el : op->input->tensorMap()) {
    inputIndices.insert(el.first);
  }
  std::set<int> outputIndices;
  for (const auto &el : op->output->tensorMap()) {
    outputIndices.insert(el.first);
  }

  for (const auto &el : identityInputToOutputIndiciesMapping) {
    InIndex inIndex = el.first;
    if (inputIndices.count(inIndex) == 0) {
      throw error(
          "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
          "identityInputToOutputIndiciesMapping has invalid input index {} "
          "for operator {}.",
          inIndex,
          op->str());
    }

    OutIndex outIndex = el.second;
    if (outputIndices.count(outIndex) == 0) {
      throw error(
          "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
          "identityInputToOutputIndiciesMapping has invalid output index {} "
          "for operator {}.",
          outIndex,
          op->str());
    }

    auto tensorIn  = op->input->tensorMap().at(inIndex);
    auto tensorOut = op->output->tensorMap().at(outIndex);

    Shape shapeIn  = tensorIn->info.shape();
    Shape shapeOut = tensorOut->info.shape();
    if (shapeIn != shapeOut) {
      throw error(
          "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
          "identityInputToOutputIndiciesMapping. You can not connect input "
          "and output with different shapes."
          "Shape of input {} is {} and "
          "Shape of output {} is {} of operator {}.",
          inIndex,
          shapeIn,
          outIndex,
          shapeOut,
          op->str());
    }

    DataType dataTypeIn  = tensorIn->info.dataType();
    DataType dataTypeOut = tensorOut->info.dataType();
    if (dataTypeIn != dataTypeOut) {
      throw error(
          "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
          "identityInputToOutputIndiciesMapping. You can not connect input "
          "and output with different data types."
          "Data types of input {} is {} and "
          "data types of output {} is {} of operator {}.",
          inIndex,
          dataTypeIn,
          outIndex,
          dataTypeOut,
          op->str());
    }
  }
}

void checkOutputIndiciesAndValues(
    const std::map<OutIndex, float> &outputIndiciesAndValues,
    Op *op) {

  std::set<int> outputIndices;
  for (const auto &el : op->output->tensorMap()) {
    outputIndices.insert(el.first);
  }

  for (const auto &el : outputIndiciesAndValues) {
    OutIndex outIndex = el.first;
    if (outputIndices.count(outIndex) == 0) {
      throw error("[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
                  "outputIndiciesAndValues has invalid output index {} "
                  "for operator {}.",
                  outIndex,
                  op->str());
    }
  }
}

void checkNM(unsigned n, unsigned m, Op *op, const Ir &ir) {

  if (n > m) {
    throw error("[AutomaticLossScale transform][[executeOpNTimesEveryMTimes]."
                "Arguments N and M of executeOpNTimesEveryMTimes are {}, {} . "
                "N must not be larger than M.",
                n,
                m);
  }

  if (op->settings.executionContext == ExecutionContext::Normal &&
      ir.getSessionOptions().enableGradientAccumulation) {
    if (ir.getSessionOptions().accumulationFactor % m != 0) {
      throw error(
          "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes]."
          "Argument M of executeOpNTimesEveryMTimes has inconsistent value {}. "
          "Operation {} is in the Normal execution context and "
          "gradient accumulation is enabled hence M should be a factor of "
          "gradient accumulation factor {}.",
          m,
          op->str(),
          ir.getSessionOptions().accumulationFactor);
    }
  } else {
    if (ir.getDataFlow().batchesPerStep() % m != 0) {
      throw error(
          "[AutomaticLossScaletransform][[executeOpNTimesEveryMTimes]. "
          "Argument M of executeOpNTimesEveryMTimes has inconsistent value {}. "
          "M should be a factor of batches per step {}.",
          m,
          ir.getDataFlow().batchesPerStep());
    }
  }
}

} // namespace

std::size_t AutomaticLossScale::id() {
  return typeid(AutomaticLossScale).hash_code();
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
    ir.addAdditionalModelProtoTensor(tensorId);
  }
  return graph.getTensors().get(tensorId);
}

// Todo: Merge with addOnesTensor
Tensor *addOneTensor(Graph &graph, const TensorId &tensorId, int32_t value) {
  auto &ir = graph.getIr();

  if (ir.tensorExistsInInitialisers(tensorId)) {
    auto tp = onnxutil::getTensorProto(ir.getModel(), tensorId);
    graph.getTensors().addVarInit(tensorId, &tp);
  } else {
    std::vector<int32_t> d(1, value);
    graph.getTensors().addVarInit(
        tensorId, TensorInfo{DataType::INT32, {}}, d.data());

    logging::transform::trace("[AutomaticLossScale transform]:"
                              "Creating Ones Tensor {}",
                              tensorId);

    ir.addAdditionalModelProtoTensor(tensorId);
  }

  return graph.getTensors().get(tensorId);
}

Op *AutomaticLossScale::executeOpNTimesEveryMTimes(
    Op *op,
    unsigned n,
    unsigned m,
    const std::map<InIndex, OutIndex> &identityInputToOutputIndiciesMapping,
    const std::map<OutIndex, float> &outputIndiciesAndValues) {
  Graph &graph = op->getGraph();
  auto &ir     = graph.getIr();
  checkNM(n, m, op, ir);
  checkFrequencyModifierSettings(
      identityInputToOutputIndiciesMapping, outputIndiciesAndValues, op->str());
  checkIdentityInputToOutputIndiciesMapping(
      identityInputToOutputIndiciesMapping, op);
  checkOutputIndiciesAndValues(outputIndiciesAndValues, op);
  checkOutputConsumers(op);
  if (n == m) {
    popart::logging::debug(
        "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes] "
        "Arguments N and M of executeOpNTimesEveryMTimes have the same value.");
    return op;
  }

  TensorId counterId = op->str() + "_counter";
  Tensor *counter    = addOneTensor(graph, counterId, -1);

  auto incrementModInplaceOp =
      graph.createOp<IncrementModInplaceOp>(1, m, Op::Settings(graph, ""));
  incrementModInplaceOp->connectInTensor(IncrementModInplaceOp::getInIndex(),
                                         counter->id);
  incrementModInplaceOp->createAndConnectOutTensor(
      IncrementModInplaceOp::getOutIndex(),
      ir.createIntermediateTensorId(counter->id));
  incrementModInplaceOp->setup();

  TensorId lessId               = op->str() + "_less";
  std::vector<int32_t> lessData = {static_cast<int32_t>(n)};
  graph.getTensors().addConstInit(
      lessId, {DataType::INT32, {}}, lessData.data());

  auto lessOp =
      graph.createOp<LessOp>(Onnx::Operators::Less_9, Op::Settings(graph, ""));
  lessOp->connectInTensor(
      LessOp::getArg0InIndex(),
      incrementModInplaceOp->outTensor(IncrementModInplaceOp::getOutIndex())
          ->id);
  lessOp->connectInTensor(LessOp::getArg1InIndex(), lessId);
  lessOp->createAndConnectOutTensor(
      LessOp::getOutIndex(),
      ir.createIntermediateTensorId(
          incrementModInplaceOp->outTensor(IncrementModInplaceOp::getOutIndex())
              ->id));
  lessOp->setup();

  std::vector<OpId> opToReplace  = {op->id};
  SubgraphableOpCluster instance = SubgraphableOpCluster(opToReplace, &graph);

  std::vector<SubgraphableOpCluster> instances = {instance};

  std::map<Op *, int> index_map_subgraph;
  Graph &subgraph = SubgraphOutline::createSubgraph(
      instances, ir, index_map_subgraph, "ComputeSubgraph");

  std::map<Op *, int> index_map_empty_subgraph;
  Graph &emptySubgraph =
      SubgraphOutline::createEmptySubgraph(instance,
                                           ir,
                                           index_map_empty_subgraph,
                                           "EmptySubgraph",
                                           identityInputToOutputIndiciesMapping,
                                           outputIndiciesAndValues);

  AliasesMap aliasesMap{&ir};

  Tensor *flag = lessOp->outTensor(LessOp::getOutIndex());
  op           = SubgraphOutline::replaceWithEmptyElseBranchIfOp(
      instance, subgraph, emptySubgraph, index_map_subgraph, aliasesMap, flag);

  return op;
}

bool AutomaticLossScale::apply(Graph &graph) const {
  // Some checks:
  // 1. Must be a training session
  // 2. The optimizer's loss scaling is non-const
  // 3. Not compatible with continuous pipelining

  AliasModel aliasModel;
  AliasModelGrower aliasModelGrower{aliasModel};
  aliasModelGrower.growFullGraph(graph, DataDependenciesOnly::Yes);

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
    histogramOp->inheritPlacementAttributes(false, aliasModel);
    histogramOp->setup();
    histogramOutputs.push_back(
        histogramOp->outTensor(HistogramOp::getOutIndex()));

    if (ir.getSessionOptions().shouldDelayVarUpdates() &&
        ir.getSessionOptions().scheduleNonWeightUpdateGradientConsumersEarly) {
      histogramOp->settings.schedulePriority =
          std::numeric_limits<double>::max();
    }
  }

  // Get the loss scale tensor and the inverse loss scale tensor:
  // the tensors to be updated
  Tensor *lossScaleTensor = getLossScaleTensor(graph);
  std::set<Tensor *> inverseLossScaleTensors =
      getInverseLossScaleTensors(graph);

  // The inverse loss scale tensor is always fp32. If the loss scale factor is
  // fp16, then we must ensure the algorithm does not increase the final loss
  // scale to above max(fp16). Othwerwise the gradient scaling and unscaling
  // will not be result in an identity operation.
  bool clipOutput = lossScaleTensor->info.dataType() == DataType::FLOAT16;

  // Pass loss scale tensor and HistogramOp outputs into the LossScaleUpdateOp
  auto lossScaleUpdateOp =
      graph.createOp<LossScaleUpdateOp>(Onnx::CustomOperators::LossScaleUpdate,
                                        lossScaleTensor->info.dataType(),
                                        clipOutput,
                                        Op::Settings(graph, ""));

  // Case 0, 1, 2 or 3: Sum the statistics tensors
  // Sum the histogram tensors
  auto statsSumOp =
      graph.createOp<SumOp>(Onnx::Operators::Sum_8, Op::Settings(graph, ""));

  for (int i = 0; i < histogramOutputs.size(); i++) {
    Tensor *tensor = histogramOutputs.at(i);
    statsSumOp->connectInTensor(i, tensor->id);
  }

  statsSumOp->createAndConnectOutTensor(SumOp::getOutIndex(),
                                        "summedHistograms");
  statsSumOp->inheritPlacementAttributes(false, aliasModel);
  statsSumOp->setup();

  // Cast the summed gradients to prevent uint32 overflow after gradient
  // accumulation and replicated all reduce.
  auto statsCastOp =
      graph.createOp<CastOp>(Onnx::Operators::Cast_9,
                             DataType::FLOAT,
                             Op::Settings(graph, "HistogramCast"));
  statsCastOp->connectInTensor(CastOp::getInIndex(), "summedHistograms");
  statsCastOp->createAndConnectOutTensor(CastOp::getOutIndex(),
                                         "summedCastedHistograms");
  statsCastOp->inheritPlacementAttributes(false, aliasModel);
  statsCastOp->setup();

  // The grad statistics tensors to connect to the LossScaleUpdateOp
  Tensor *finalStatisticsTensor = statsCastOp->outTensor(CastOp::getOutIndex());

  // Case 2 or 3, Accumulate the summed gradient statistics in the inner loop,
  // reset the accumulator in the outer loop.
  if (doingGradientAccumulation(graph)) {
    // 1. add variable accl tensor, set tensor data to zeros
    TensorId toAcclId = getStatisticsAccumulatorTensorId();
    std::vector<uint32_t> d(finalStatisticsTensor->info.nelms(), 0);
    graph.getTensors().addVarInit(
        toAcclId, finalStatisticsTensor->info, d.data());

    // 2. add op to accumulate
    TensorId inId    = finalStatisticsTensor->id;
    auto statsAcclOp = graph.createOp<AddRhsInplaceOp>(Op::Settings(graph, ""));
    statsAcclOp->connectInTensor(AddRhsInplaceOp::getArg0InIndex(), inId);
    statsAcclOp->connectInTensor(AddRhsInplaceOp::getArg1InIndex(), toAcclId);
    statsAcclOp->createAndConnectOutTensor(AddRhsInplaceOp::getOutIndex(),
                                           inId + "_accld");
    statsAcclOp->inheritPlacementAttributes(false, aliasModel);
    statsAcclOp->setup();
    TensorId accldId = statsAcclOp->outId(AddRhsInplaceOp::getOutIndex());

    // 3. add AccumulatorZeroOp to reset the accl tensor
    auto statsAcclResetOp =
        graph.createOp<AccumulatorZeroOp>(Op::Settings(graph, ""));
    statsAcclResetOp->connectInTensor(
        AccumulatorZeroOp::getVarToUpdateInIndex(), toAcclId);
    statsAcclResetOp->createAndConnectOutTensor(
        AccumulatorZeroOp::getUpdatedVarOutIndex(), inId + "_accld_reset");
    statsAcclResetOp->inheritPlacementAttributes(false, aliasModel);
    statsAcclResetOp->setup();

    // Reset the accumulator, and update the loss in the outer fragment
    statsAcclResetOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;

    graph.topoCons->insert(lossScaleUpdateOp, statsAcclResetOp);

    finalStatisticsTensor =
        statsAcclOp->outTensor(AddRhsInplaceOp::getOutIndex());
  }

  // Case 1 or 3, Reduce the summed or accumulated statistics tensor
  if (doingGraphReplication(graph)) {
    TensorId inId = finalStatisticsTensor->id;

    auto reduceOp = graph.createOp<ReplicatedAllReduceOp>(
        Onnx::CustomOperators::ReplicatedAllReduce,
        Op::Settings(graph, inId + "_reduced"));

    reduceOp->connectInTensor(ReplicatedAllReduceOp::getInIndex(), inId);
    reduceOp->createAndConnectOutTensor(ReplicatedAllReduceOp::getOutIndex(),
                                        "summedHistograms_reduced");
    reduceOp->inheritPlacementAttributes(false, aliasModel);
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

    finalStatisticsTensor =
        reduceOp->outTensor(ReplicatedAllReduceOp::getOutIndex());
  }

  lossScaleUpdateOp->connectInTensor(
      LossScaleUpdateOp::getStatisticsTensorInIndex(),
      finalStatisticsTensor->id);

  // Create variable tensor, initialised to 'ones' for input at
  // LossScaleUpdateOp::getLossScaleUpdateFactorInIndex
  Tensor *lsUpdateFactor = addOnesTensor(
      graph, getLossScaleUpdateFactorTensorId(), lossScaleTensor->info);

  lossScaleUpdateOp->connectInTensor(
      LossScaleUpdateOp::getLossScaleUpdateFactorInIndex(), lsUpdateFactor->id);
  lossScaleUpdateOp->connectInTensor(LossScaleUpdateOp::getLossScalingInIndex(),
                                     lossScaleTensor->id);
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
    if (consumer != lsMulOp && consumer != lossScaleUpdateOp) {
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
