// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <onnxutil.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/aliasesmap.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulatorzero.hpp>
#include <popart/op/add.hpp>
#include <popart/op/autolossscaleproxy.hpp>
#include <popart/op/cast.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/convbase.hpp>
#include <popart/op/copyvarupdate.hpp>
#include <popart/op/div.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/histogram.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/incrementmod.hpp>
#include <popart/op/less.hpp>
#include <popart/op/lossscaleupdate.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/slice.hpp>
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

bool producesOutputsOfTypeFp16(const Op *op) {
  for (Tensor *tensor : op->output->tensors()) {
    if (tensor->info.dataType() != DataType::FLOAT16) {
      return false;
    }
  }
  return true;
}

int numberOfConsumers(const Op *op) {
  std::set<Op *> consumers;
  for (Tensor *tensor : op->output->tensors()) {
    for (Op *op : tensor->consumers.getOps()) {
      consumers.insert(op);
    }
  }

  return consumers.size();
}

bool consumesVariableInput(const Op *op) {
  for (Tensor *tensor : op->input->tensors()) {
    if (tensor->tensorType() == TensorType::Variable) {
      return true;
    }
  }
  return false;
}

bool producesConvOrMatmulGradient(const Op *op) {
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

bool producesNonViewChangingGradientTensor(const Op *op) {
  if (op->fromLoss != PathFromLoss::Yes) {
    return false;
  }
  if (op->isInplaceViewChange() || op->isOutplaceViewChange()) {
    return false;
  }
  if (!producesOutputsOfTypeFp16(op)) {
    return false;
  }
  if (op->settings.executionContext != ExecutionContext::Normal) {
    return false;
  }
  // Prevent cycles in the graph
  if (consumesVariableInput(op)) {
    return false;
  }
  // Doesn't lead to a VarUpdateOp, so we shouldn't care about its statistics
  if (numberOfConsumers(op) == 0) {
    return false;
  }
  // Non-view-changing ops, but ones that will definitely not cause overflow.
  if (op->isConvertibleTo<ConcatOp>() || op->isConvertibleTo<ConcatGradOp>() ||
      op->isConvertibleTo<GatherOp>() || op->isConvertibleTo<GatherGradOp>() ||
      op->isConvertibleTo<SliceOp>() || op->isConvertibleTo<SliceGradOp>()) {
    return false;
  }

  return true;
}

void removeProxyOps(Graph &graph) {
  auto &opIdOpMap = graph.getOps();
  // We would like to delete contents of the map while looping through, hence
  // the following loop
  for (auto opIdOpMapIt = opIdOpMap.begin(); opIdOpMapIt != opIdOpMap.end();
       /* Increment happens in body */) {
    auto op = opIdOpMapIt->second.get();
    if (op->opid.type == "AutoLossScaleProxy") {
      auto tensorIn  = op->input->tensors()[0];
      auto tensorOut = op->output->tensors()[0];

      for (auto &consumer : tensorOut->consumers.getOps()) {
        std::vector<int> indices = consumer->input->indicesMap().at(tensorOut);
        for (auto i : indices) {
          consumer->disconnectInTensor(i, tensorOut);
          consumer->connectInTensor(i, tensorIn->id);
        }
      }

      op->disconnectAllInputs();
      op->disconnectAllOutputs();

      // Returns iterator following the last removed element
      opIdOpMapIt = graph.eraseOp(op->id);
    } else {
      ++opIdOpMapIt;
    }
  }
}

void removeProxyGradOps(Graph &graph) {
  auto &opIdOpMap = graph.getOps();
  // We would like to delete contents of the map while looping through, hence
  // the following loop
  for (auto opIdOpMapIt = opIdOpMap.begin(); opIdOpMapIt != opIdOpMap.end();
       /* Increment happens in body */) {
    auto op = opIdOpMapIt->second.get();
    if (op->opid.type == "AutoLossScaleProxyGrad") {
      auto tensorIn  = op->input->tensors()[0];
      auto tensorOut = op->output->tensors()[0];

      op->disconnectAllInputs();
      op->disconnectAllOutputs();

      Op *producer             = tensorIn->getProducer();
      std::vector<int> indices = producer->output->indicesMap().at(tensorIn);
      producer->disconnectOutTensor(tensorIn);
      for (auto i : indices) {
        producer->connectOutTensor(i, tensorOut->id);
      }

      // Returns iterator following the last removed element
      opIdOpMapIt = graph.eraseOp(op->id);
    } else {
      ++opIdOpMapIt;
    }
  }
}

// Users can provide names of forward tensors in order to their
// gradient tensors were used for automatic loss scaling.
// They are temporarily annotated in graphs by AutoLossScaleProxyOp
// to find their gradient tensors after automatic differentiation.
// We have to remove the AutoLossScaleProxyOps and AutoLossScaleProxyGradOps.
// Collect the gradient tensors.
std::vector<Tensor *> getToTrackTensorsUser(Graph &graph) {
  std::vector<Tensor *> toTrackTensors;

  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (op->opid.type == "AutoLossScaleProxyGrad") {
      for (Tensor *tensor : op->output->tensors()) {
        toTrackTensors.push_back(tensor);
      }
    }
  }

  removeProxyOps(graph);
  removeProxyGradOps(graph);

  return toTrackTensors;
}

std::vector<Tensor *> getToTrackTensorsAuto(Graph &graph) {
  auto settings =
      graph.getIr().getSessionOptions().automaticLossScalingSettings;
  auto producesToTrackTensors_ = [&settings](const Op *op) {
    if (settings.gradientTensorTrackingMethod ==
        GradientTensorTrackingMethod::ConvAndMatmulGradients) {
      return producesConvOrMatmulGradient(op);
    } else if (settings.gradientTensorTrackingMethod ==
               GradientTensorTrackingMethod::
                   AllNonViewChangingGradientTensors) {
      return producesNonViewChangingGradientTensor(op);
    } else {
      throw internal_error("[AutomaticLossScale transform] Unknown "
                           "GradientTensorTrackingMethod.");
    }
  };

  std::vector<Tensor *> toTrackTensors;

  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();

    if (producesToTrackTensors_(op)) {
      for (Tensor *tensor : op->output->tensors()) {
        // In the case where gradient tensors of type float32 and float16 are
        // present in the list of candidate 'ToTrackTensors', filter out the
        // float32 gradients. They should not influence the loss scaling
        // factor. Keep only float16 tensors.
        if (tensor->info.dataType() == DataType::FLOAT16) {
          toTrackTensors.push_back(tensor);
        }
      }
    }
  }

  // Verify that we are returning a non-empty vector of to-track tensors
  if (toTrackTensors.size() == 0) {
    throw error("[AutomaticLossScale transform] No tracked gradient tensors of "
                "type fp16 were found.");
  }

  return toTrackTensors;
}

std::vector<Tensor *> getToTrackTensors(Graph &graph) {
  auto settings =
      graph.getIr().getSessionOptions().automaticLossScalingSettings;
  if (settings.toTrackTensors.has_value() &&
      settings.gradientTensorTrackingMethod !=
          GradientTensorTrackingMethod::GradientsOfUserSpecifiedTensors) {
    throw error(
        "[AutomaticLossScale transform] toTrackTensors has been set, but "
        "gradientTensorTrackingMethod has not been set to "
        "'GradientTensorTrackingMethod::GradientsOfUserSpecifiedTensors'.");
  }
  if (settings.gradientTensorTrackingMethod ==
      GradientTensorTrackingMethod::GradientsOfUserSpecifiedTensors) {
    if (!settings.toTrackTensors.has_value()) {
      throw error(
          "[AutomaticLossScale transform] gradientTensorTrackingMethod set to "
          "'GradientTensorTrackingMethod::GradientsOfUserSpecifiedTensors' but "
          "'toTrackTensors' has not been set");
    } else {
      return getToTrackTensorsUser(graph);
    }
  } else {
    return getToTrackTensorsAuto(graph);
  }
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
  if (dtype == DataType::FLOAT16) {
    // TODO: T54890 don't use magic numbers. Get max value of IeeeHalf
    // programatically.
    return {static_cast<float>(65504.0) * binEdgeLocation};
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
    if (m % ir.getSessionOptions().accumulationFactor != 0 &&
        ir.getSessionOptions().accumulationFactor % m != 0) {
      throw error(
          "[AutomaticLossScale transform][[executeOpNTimesEveryMTimes]. "
          "Argument M of executeOpNTimesEveryMTimes has inconsistent value {}. "
          "Operation {} is in the Normal execution context and "
          "gradient accumulation is enabled hence M should be a factor "
          "or multiple of gradient accumulation factor {}.",
          m,
          op->str(),
          ir.getSessionOptions().accumulationFactor);
    }
  }
}

Op *executeHistogramNTimesEveryMTimes(Op *op,
                                      const Ir &ir,
                                      AliasModel aliasMode) {

  int updatePeriod =
      ir.getSessionOptions().automaticLossScalingSettings.updatePeriod;
  unsigned n;
  unsigned m;
  if (ir.getSessionOptions().enableGradientAccumulation) {
    int accumFactor = ir.getSessionOptions().accumulationFactor;
    n               = accumFactor;
    m               = accumFactor * updatePeriod;

  } else {
    n = 1;
    m = updatePeriod;
  }

  std::map<InIndex, OutIndex> identityInputToOutputIndiciesMapping;
  std::map<OutIndex, float> outputIndiciesAndValues{{0, 0}};

  return AutomaticLossScale::executeOpNTimesEveryMTimes(
      op,
      n,
      m,
      identityInputToOutputIndiciesMapping,
      outputIndiciesAndValues,
      aliasMode);
}

Op *executeLossScaleUpdateNTimesEveryMTimes(Op *op,
                                            int updatePeriod,
                                            AliasModel aliasMode) {

  std::map<InIndex, OutIndex> identityInputToOutputIndiciesMapping{{0, 0}};
  std::map<OutIndex, float> outputIndiciesAndValues;

  return AutomaticLossScale::executeOpNTimesEveryMTimes(
      op,
      1,
      updatePeriod,
      identityInputToOutputIndiciesMapping,
      outputIndiciesAndValues,
      aliasMode);
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
    const std::map<OutIndex, float> &outputIndiciesAndValues,
    AliasModel &aliasModel) {
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
  incrementModInplaceOp->inheritPlacementAttributes(false, aliasModel);
  incrementModInplaceOp->setVirtualGraphId(op->getOptionalVGraphId());
  if (op->settings.executionContext ==
      ExecutionContext::AccumulateOuterFragment) {
    incrementModInplaceOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  } else if (op->settings.executionContext == ExecutionContext::Normal) {
    if (op->hasPipelineStage()) {
      incrementModInplaceOp->setPipelineStage(op->getOptionalPipelineStage());
    }
  }

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
  lessOp->inheritPlacementAttributes(false, aliasModel);
  if (op->settings.executionContext ==
      ExecutionContext::AccumulateOuterFragment) {
    lessOp->settings.executionContext =
        ExecutionContext::AccumulateOuterFragment;
  } else if (op->settings.executionContext == ExecutionContext::Normal) {
    if (op->hasPipelineStage()) {
      lessOp->setPipelineStage(op->getOptionalPipelineStage());
    }
  }
  lessOp->setup();

  std::vector<OpId> opToReplace  = {op->id};
  SubgraphableOpCluster instance = SubgraphableOpCluster(opToReplace, &graph);

  std::vector<SubgraphableOpCluster> instances = {instance};

  std::map<Op *, int> index_map_subgraph;
  Graph &subgraph = SubgraphOutline::createSubgraph(
      instances, ir, index_map_subgraph, "ComputeSubgraph");

  std::string subgraphId = op->str() + "_EmptySubgraph";
  Graph &emptySubgraph =
      SubgraphOutline::createEmptySubgraph(instance,
                                           ir,
                                           subgraphId,
                                           identityInputToOutputIndiciesMapping,
                                           outputIndiciesAndValues,
                                           aliasModel);

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
  std::vector<Op *> histograms;
  for (Tensor *tensor : toTrackTensors) {
    logging::transform::debug("Collecting statistics for tensor '{}' for "
                              "control of loss-scale value.",
                              tensor->id);

    // Get automatic loss scaling hyperparameters.
    float binEdgeLocation =
        ir.getSessionOptions().automaticLossScalingSettings.binEdgeLocation;

    // Attach a newly created HistogramOp to each tensor
    Op *histogramOp =
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
    histograms.push_back(histogramOp);
  }

  // Get the loss scale tensor and the inverse loss scale tensor:
  // the tensors to be updated
  Tensor *lossScaleTensor = getLossScaleTensor(graph);
  std::set<Tensor *> inverseLossScaleTensors =
      getInverseLossScaleTensors(graph);

  // Pass loss scale tensor and HistogramOp outputs into the LossScaleUpdateOp
  Op *lossScaleUpdateOp =
      graph.createOp<LossScaleUpdateOp>(Onnx::CustomOperators::LossScaleUpdate,
                                        lossScaleTensor->info.dataType(),
                                        Op::Settings(graph, "LossScaleUpdate"));

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

  int updatePeriod =
      ir.getSessionOptions().automaticLossScalingSettings.updatePeriod;

  Op *copyVarUpdates;
  if (updatePeriod > 1) {
    auto copyVarUpdateOp =
        graph.createOp<CopyVarUpdateOp>(Op::Settings(graph, ""));
    copyVarUpdateOp->connectInTensor(CopyVarUpdateOp::getVarToUpdateInIndex(),
                                     lsUpdateFactor->id);
    copyVarUpdateOp->connectInTensor(
        CopyVarUpdateOp::getUpdaterInIndex(),
        lossScaleUpdateOp
            ->outTensor(
                LossScaleUpdateOp::getUpdatedLossScaleUpdateFactorOutIndex())
            ->id);
    copyVarUpdateOp->createAndConnectOutTensor(
        CopyVarUpdateOp::getUpdatedVarOutIndex(),
        lsUpdateFactor->id + "_updated");
    copyVarUpdateOp->inheritPlacementAttributes(false, aliasModel);
    copyVarUpdateOp->setup();
    copyVarUpdateOp->pruneable = false;
    copyVarUpdates             = copyVarUpdateOp;
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

  // When updatePeriod is larger than one we apply executeOpNTimesEveryMTimes
  // tool on histogram ops and on loss scale update op which reduces
  // the frequency of computations of these ops,
  // with the intended outcome of improving throughput.
  if (updatePeriod > 1) {
    copyVarUpdates->setVirtualGraphId(vgId);

    for (Op *histogramOp : histograms) {
      histogramOp =
          executeHistogramNTimesEveryMTimes(histogramOp, ir, aliasModel);
    }

    lossScaleUpdateOp = executeLossScaleUpdateNTimesEveryMTimes(
        lossScaleUpdateOp, updatePeriod, aliasModel);
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new AutomaticLossScale);
}

} // namespace popart
