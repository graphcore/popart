// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/random/normal_distribution.hpp>

#include <popart/builder.hpp>
#include <popart/ces/constexpr.hpp>
#include <popart/ces/onnxconstexpr.hpp>
#include <popart/chains.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/intervals.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/init.hpp>
#include <popart/op/loss.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/pbwrap.hpp>
#include <popart/scheduler.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>

// The transformations
#include <popart/recompute.hpp>
#include <popart/transforms/auto_virtual_graph.hpp>
#include <popart/transforms/batchserialize.hpp>
#include <popart/transforms/cachesetup.hpp>
#include <popart/transforms/decomposegradsum.hpp>
#include <popart/transforms/devicereduce.hpp>
#include <popart/transforms/dynamicoptransform.hpp>
#include <popart/transforms/explicitrecompute.hpp>
#include <popart/transforms/groupmatmuls.hpp>
#include <popart/transforms/hostreduce.hpp>
#include <popart/transforms/inferpipelinestages.hpp>
#include <popart/transforms/interipucopy.hpp>
#include <popart/transforms/iocomputetilecopy.hpp>
#include <popart/transforms/mergecopies.hpp>
#include <popart/transforms/mergeduplicateops.hpp>
#include <popart/transforms/mergevarupdates.hpp>
#include <popart/transforms/pingpong.hpp>
#include <popart/transforms/pipeline.hpp>
#include <popart/transforms/prune.hpp>
#include <popart/transforms/serializematmuls.hpp>
#include <popart/transforms/subgraphoutline.hpp>

// The layers required to construct the backwards pass
#include <popart/op/batchnorm.hpp>
#include <popart/op/copyvarupdate.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/placeholder.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/sum.hpp>

#include <popart/patterns/inplace.hpp>
#include <popart/patterns/sgd1decompose.hpp>
#include <popart/patterns/updateinplaceprioritiesforipu.hpp>

#include <popart/dotvisualizer.hpp>

// used for float to half conversion
#include <poplar/Target.hpp>

namespace popart {

Ir::~Ir() = default;

void Ir::confirmNonReservedId(const TensorId &tenId) const {
  for (auto reservedPrefix : reservedPrefixes()) {
    if (tenId.find(reservedPrefix) != std::string::npos) {
      throw error("Provided tensor " + tenId +
                  " has an invalid name: clash with reserved prefix " +
                  reservedPrefix);
    }
  }
}

GradNonGradPair::GradNonGradPair(Op *g_, Op *ng_) : grad(g_), nongrad(ng_) {}

GradNonGradPair::GradNonGradPair() : GradNonGradPair(nullptr, nullptr) {}

const ONNX_NAMESPACE::ModelProto &Ir::getModel() const { return *onnxModel; }

// Data stream tensors are all tensors, excluding:
//  - optimizer tensors
//  - the random seed tensor
std::vector<Tensor *> Ir::dataStreamTensors() const {
  std::vector<Tensor *> dsTensors;
  for (auto tensor : getTensors().getOfType(TensorType::Stream)) {
    if (!tensor->isOptimizerTensor()) {
      if (!tensor->isRandomSeedTensor()) {
        dsTensors.push_back(tensor);
      }
    }
  }
  return dsTensors;
}

std::vector<Tensor *> Ir::optimizerTensors() const {
  std::vector<Tensor *> optimizerTensors;
  for (auto tensor : getTensors().getOfType(TensorType::Stream)) {
    if (tensor->isOptimizerTensor()) {
      optimizerTensors.push_back(tensor);
    }
  }
  return optimizerTensors;
}

void Ir::updateOptimizer(const Optimizer &newOptimizer) {
  // TODO this will be cleaner when T12589 is done
  auto newOptimizerClone = newOptimizer.clone();
  newOptimizerClone->setFactorsFromOptions(getSessionOptions());
  if (!optimizer->validReplacement(*newOptimizerClone)) {
    throw error("This Optimizer of type " + newOptimizer.type_s() +
                " is not a valid replacement for optimizer of type " +
                optimizer->type_s());
  }
  optimizer = std::move(newOptimizerClone);
  for (auto opt : optimizerTensors()) {
    optimizer->resetTensorData(*opt);
  }
}

void Ir::dotCheckpoint(DotCheck check) const {
  DotVisualizer viz(this, check);
  viz.write();
}

void Ir::confirmNoReservedIds() const {

  auto &onnxGraph = onnxModel->graph();

  for (const auto &in_ : onnxGraph.input()) {
    confirmNonReservedId(in_.name());
  }

  for (const auto &out_ : onnxGraph.output()) {
    confirmNonReservedId(out_.name());
  }

  for (const auto &tenId : inputShapeInfo.getAllTensorIds()) {
    confirmNonReservedId(tenId);
  }
}

IrBundle::IrBundle(const ONNX_NAMESPACE::ModelProto &modelProto_,
                   const InputShapeInfo &inputShapeInfo_,
                   const DataFlow &dataFlow_,
                   const TensorId &loss_,
                   const Optimizer *optimizer_,
                   DeviceInfo &deviceInfo_,
                   const SessionOptions &userOptions_,
                   const Patterns &patterns_)
    : modelProto(modelProto_), inputShapeInfo(inputShapeInfo_),
      dataFlow(dataFlow_), loss(loss_), optimizer(optimizer_),
      deviceInfo(deviceInfo_), userOptions(userOptions_), patterns(patterns_) {}

Ir::Ir() : onnxModel(nullptr) {
  graphs.insert(
      {GraphId::root(), std::make_unique<Graph>(*this, GraphId::root())});
}

void Ir::setOnnxModel(const ONNX_NAMESPACE::ModelProto &model) {
  onnxModel.reset(new ONNX_NAMESPACE::ModelProto(model));
}

void Ir::setDataFlow(const DataFlow &df) {
  // Inference  mode require an anchor
  if (!canTrain() && df.nAnchors() == 0) {
    throw error("User must specify an anchor tensor when doing inference.");
  } else {
    dataFlow = df;
  }
}

bool Ir::virtualGraphsEnabled() const {
  return userOptions.virtualGraphMode != VirtualGraphMode::Off;
}

SyntheticDataMode Ir::syntheticDataMode() const {
  return getSessionOptions().syntheticDataMode;
}

bool Ir::useSyntheticData() const {
  return syntheticDataMode() != SyntheticDataMode::Off;
}

void Ir::setUserOptions(const SessionOptions &flags) { userOptions = flags; }

void Ir::setInputShapeInfo(const InputShapeInfo &info) {
  inputShapeInfo = info;
}

void Ir::setPatterns(const Patterns &p) {
  logging::pattern::info("Enabling {} patterns", getPatternLevelStr(p));
  patterns = p;
}

std::string Ir::getPatternLevelStr(const Patterns &p) {
  if (isPatternsLevel(p, PatternsLevel::All)) {
    return "all";
  } else if (isPatternsLevel(p, PatternsLevel::Default)) {
    return "default";
  } else if (isPatternsLevel(p, PatternsLevel::NoPatterns)) {
    return "no";
  } else {
    return "custom";
  }
}

bool Ir::isPatternsLevel(const Patterns &p, PatternsLevel level) {
  Patterns refPatterns(level);
  if (refPatterns == p) {
    return true;
  } else {
    return false;
  }
}

void Ir::removeIsolatedTensors(bool retainCached) {
  getTensors().removeIsolated(retainCached);
}

void Ir::setExecutionMode(const ExecutionMode &mode) { executionMode = mode; }

void Ir::setOptimizer(const Optimizer &o) {
  optimizer = o.clone();
  optimizer->setFactorsFromOptions(getSessionOptions());

  // We create scale factor Tensors now (they will be removed later if not
  // used). All other optimizer Tensors are created just-in-time during Graph
  // construction
  for (DataType dt : {DataType::FLOAT, DataType::FLOAT16}) {
    auto id = optimizer->getLossScalingTensorId(dt);
    ensureOptimizerTensorCreated(id, {dt, {}});
  }
}

void Ir::setDeviceInfo(DeviceInfo &di) { deviceInfo = &di; }

const DeviceInfo *Ir::getDeviceInfo() const { return deviceInfo; }

void Ir::logIr() {
  std::stringstream ss2;
  append(ss2);
  logging::ir::info(ss2.str());
}

void Ir::verifyPipelineSettings() const {
  if (!getSessionOptions().enablePipelining) {
    // If pipelining is disabled, make sure no ops have a pipeline stage set.
    for (auto &id_graph : graphs) {
      auto &graph = id_graph.second;
      for (auto &id_op : graph->getOps()) {
        auto &op = id_op.second;
        // no pipeline stage
        op->setPipelineStage({});
      }
    }

    return;
  }

  if (!virtualGraphsEnabled() || getMaxVirtualGraphId() == 1) {
    throw error("Pipelining requires more than 1 IPU and the "
                "'virtualGraphMode' session option "
                "to not be VirtualGraphMode::Off.");
  }

  auto getPipelineStage = [](auto x) -> PipelineStage {
    if (x->hasPipelineStage()) {
      return x->getPipelineStage();
    } else {
      return -1;
    }
  };

  auto getVirtualGraphId = [](auto x) -> VGraphId {
    if (x->hasVirtualGraphId()) {
      return x->getVirtualGraphId();
    } else {
      return -1;
    }
  };

  // collect a set of vgraph ids for each pipeline stage
  std::map<PipelineStage, std::vector<Op *>> pipelineStages;
  std::map<VGraphId, std::set<PipelineStage>> pipelineStagesPerVGraph;

  for (auto &id_op : getMainGraph().getOps()) {
    auto op = id_op.second.get();
    if (!op->isConvertibleTo<IpuCopyOp>()) {
      auto ps = getPipelineStage(op);
      pipelineStages[ps].push_back(op);

      auto vgraph = getVirtualGraphId(op);
      pipelineStagesPerVGraph[vgraph].insert(ps);
    }
  }

  // if no ops have had the pipeline stage attribute set, the virtual graph id
  // will be used.

  // some ops have not had the pipeline stage attribute set
  if (pipelineStages.count(-1) != 0 && pipelineStages.size() > 1) {
    std::stringstream ss;
    ss << "Only some ops have had their pipeline stage set. Ops missing the "
          "pipeline stage:";
    for (auto &id_op : getMainGraph().getOps()) {
      auto op = id_op.second.get();
      if (!op->isConvertibleTo<IpuCopyOp>()) {
        if (getPipelineStage(op) == -1) {
          ss << logging::format("\n  {}", op->debugName());
        }
      }
    }
    throw error(ss.str());
  }
  // all ops have had the pipeline stage attribute set
  else if (pipelineStages.count(-1) == 0) {

    // check that all ops in a pipeline stage have the same virtualGraph
    for (auto &ps_ops : pipelineStages) {
      auto ps   = ps_ops.first;
      auto &ops = ps_ops.second;

      std::set<VGraphId> vgraphs;
      for (auto op : ops) {
        // ops may not have a virtual graph id yet as the virtualGraphMode may
        // be Auto. In this case getVirtualGraphId returns -1 and we just check
        // that all ops in the pipeline stage are on virtual graph -1
        vgraphs.insert(getVirtualGraphId(op));
      }

      if (vgraphs.size() > 1) {
        std::vector<std::string> opNames;
        for (auto op : ops) {
          opNames.push_back(op->debugName());
        }

        throw error("Ops {} have the same pipeline stage {}, but different "
                    "virtual graph ids {}. All ops with the same pipeline "
                    "stage must also have the same virtual graph id",
                    opNames,
                    ps,
                    vgraphs);
      }
    }
  }
}

void Ir::verifyPingPongSettings() const {
  // check for mismatched settings
  if (userOptions.pingPongPhases > 1 &&
      userOptions.virtualGraphMode != VirtualGraphMode::PingPong) {
    throw error("PingPong phases > 1 requires VirtualGraphMode::PingPong");
  }

  // if pingpong is enabled
  if (userOptions.virtualGraphMode == VirtualGraphMode::PingPong &&
      userOptions.pingPongPhases > 1) {
    // Currently there are no checks for when ping pong is enabled.
  } else {
    // if pingpong is disabled, make sure all ops pingpong phases are set to
    // nonstd::nullopt.
    for (auto &id_graph : graphs) {
      auto &graph = id_graph.second;
      for (auto &id_op : graph->getOps()) {
        auto &op = id_op.second;
        op->setPingPongPhase({});
      }
    }
  }
}

void Ir::verifyOpOutputConnectivity(const Graph &graph) const {
  logging::ir::debug("Checking op output tensor producers for graph '{}'",
                     graph.id.str());

  // Check op output tensor producers
  for (auto &op_pair : graph.getOps()) {
    auto &op = op_pair.second;

    for (auto &tensor_pair : op->output->tensorMap()) {
      auto t = tensor_pair.second;

      if (!t->hasProducer()) {
        throw error("Tensor {} should have a producer", t->str());
      }

      if (t->getProducer() != op.get()) {
        throw error(
            "Op {} should produce {}, but it's not the assigned producer",
            op->str(),
            t->str());
      }
    }
  }
}

void Ir::verifyOpInputConnectivity(const Graph &graph) const {
  logging::ir::debug("Checking op input tensor consumers for graph '{}'",
                     graph.id.str());

  // Count the number of times an op consumes its input tensors
  std::map<std::pair<Tensor *, Op *>, int> consumption_count;
  for (auto &op_pair : graph.getOps()) {
    auto &op = op_pair.second;

    for (auto &tensor_pair : op->input->tensorMap()) {
      auto t = tensor_pair.second;

      consumption_count[{t, op.get()}]++;
    }
  }

  // Check that the consumption count matches the value reported by Consumers::n
  for (auto &cons_count : consumption_count) {
    auto tensor = cons_count.first.first;
    auto op     = cons_count.first.second;
    auto count  = cons_count.second;

    if (tensor->consumers.n(op) != count) {
      throw error("Op {} should consume {} {} times, but it "
                  "consumes it {} times",
                  op->str(),
                  tensor->str(),
                  count,
                  tensor->consumers.n(op));
    }
  }
}

void Ir::verifyTensorProducerConnectivity() const {
  logging::ir::debug("Checking tensor producer outputs");

  for (auto &tid : getTensors().getAllTensorIds()) {
    auto tensor = getTensors().get(tid);

    if (tensor->hasProducer() && tensor->tensorType() == TensorType::Stream) {
      auto op = tensor->getProducer();
      throw error("Tensor {} is a stream tensor, but has op {} as a producer",
                  tensor->str(),
                  op->str());
    }

    if (tensor->hasProducer() && tensor->tensorType() == TensorType::Const) {
      auto op = tensor->getProducer();
      throw error("Tensor {} is a const tensor, but has op {} as a producer",
                  tensor->str(),
                  op->str());
    }

    if (tensor->hasProducer() && tensor->tensorType() == TensorType::Variable) {
      auto op = tensor->getProducer();
      if (!dynamic_cast<VarUpdateOp *>(op)) {
        throw error(
            "Tensor {} is a variable tensor, but has op {} as a producer",
            tensor->str(),
            op->str());
      }
    }

    if (!tensor->hasProducer() && tensor->tensorType() == TensorType::ActGrad) {
      throw error("Tensor {} is an actgrad tensor, but doesn't have a producer",
                  tensor->str());
    }

    // Check that the producer op has the tensor as an output
    if (tensor->hasProducer()) {
      auto op = tensor->getProducer();

      if (op->output == nullptr) {
        throw error("Op {} output tensor index map is null", op->str());
      }

      if (op->output->indices(tensor).empty()) {
        throw error(
            "Tensor {} has op {} as a producer, but it doesn't appear in "
            "the op's outputs",
            tensor->str(),
            op->str());
      }

      if (op->output->indices(tensor).size() > 1) {
        throw error("Tensor {} has op {} as a producer, but it appears in "
                    "the op's outputs {} times",
                    tensor->str(),
                    op->str(),
                    op->output->indices(tensor).size());
      }
    }
  }
}

void Ir::verifyTensorConsumerConnectivity() const {
  logging::ir::debug("Checking tensor consumer inputs");

  // Count the number of times a tensor is consumed by an op
  std::map<std::pair<Tensor *, Op *>, int> consumption_count;
  for (auto &tid : getTensors().getAllTensorIds()) {
    auto tensor = getTensors().get(tid);

    for (auto op : tensor->consumers.getOps()) {
      consumption_count[{tensor, op}] += tensor->consumers.n(op);
    }
  }

  // Check that the consumption count matches the value reported by
  // op->input->indices(tensor).size()
  for (auto &cons_count : consumption_count) {
    auto tensor = cons_count.first.first;
    auto op     = cons_count.first.second;
    auto count  = cons_count.second;

    if (op->input == nullptr) {
      throw error("Op {} input tensor index map is null", op->str());
    }

    if (op->input->indices(tensor).size() != count) {
      throw error("Tensor {} should have op {} as a consumer {} times, but it "
                  "consumes it {} times",
                  tensor->str(),
                  op->str(),
                  op->input->indices(tensor).size(),
                  count);
    }
  }
}

void Ir::verifyConnectivity() const {
  logging::ir::info("Checking IR connectivity");

  for (auto &x : graphs) {
    auto &graph = *x.second.get();
    verifyOpInputConnectivity(graph);
    verifyOpOutputConnectivity(graph);
  }
  verifyTensorProducerConnectivity();
  verifyTensorConsumerConnectivity();

  logging::ir::info("IR connectivity check passed");
}

void Ir::verifyTensorIds() const {
  logging::ir::info("Checking TensorIds are unique");

  // Check that all TensorIds are unique
  std::set<TensorId> seen;

  for (auto &id_graph : graphs) {
    auto graph = id_graph.second.get();
    for (auto &id : graph->getTensors().getAllTensorIds()) {
      if (seen.find(id) != seen.end()) {
        throw error("TensorId '{}' is not unique", id);
      } else {
        seen.insert(id);
      }
    }
  }

  logging::ir::info("TensorId check passed");
}

void Ir::verifySubgraphs() const {
  for (auto graph : getAllGraphs()) {
    // Verify that no subgraph contains an Op that is not outlinable
    if (graph->id == getMainGraph().id) {
      continue;
    }
    for (auto &id_op : graph->getOps()) {
      auto op = id_op.second.get();
      if (op->isOutlineable() == false) {
        throw error("Subgraph '{}' contains op '{}', but ops of type '{}' are "
                    "not outlineable",
                    graph->id,
                    op->debugName(),
                    op->opid);
      }
    }
  }
}

void Ir::verifyRecomputeAttributes() const noexcept(false) {
  // If explicit recomputation is turned on
  // No op is allowed to have its recompute type set to Recompute
  if (userOptions.explicitRecomputation) {
    for (auto op : getAllOps()) {
      if (op->settings.recomputeType == RecomputeType::Recompute) {
        throw error("Explicit recomputation is turned on for op '{}', but its "
                    "recompute type is set to '{}'",
                    op->debugName(),
                    op->settings.recomputeType);
      }
    }
  }
}

void Ir::verifyDistributedReplicatedGraphSettings() const {
  if (userOptions.enableDistributedReplicatedGraphs) {
    auto localReplicationFactor  = userOptions.replicatedGraphCount;
    auto globalReplicationFactor = userOptions.globalReplicationFactor;
    auto globalReplicaOffset     = userOptions.globalReplicaOffset;
    if (globalReplicationFactor < 1) {
      throw error("Invalid globalReplicationFactor value: {}, must be greater "
                  "or equal than 1",
                  globalReplicationFactor);
    }

    if (globalReplicaOffset < 0) {
      throw error("Invalid globalReplicaOffset value: {}, must be greater or "
                  "equal than 0",
                  globalReplicaOffset);
    }

    if (globalReplicaOffset > globalReplicationFactor) {
      throw error("Global replica offset: {}, is larger than global "
                  "replication factor: {}",
                  globalReplicaOffset,
                  globalReplicationFactor);
    }

    if (userOptions.enableReplicatedGraphs) {
      if (localReplicationFactor == 1) {
        throw error(
            "Local replicated graphs enabled but replication factor is 1");
      }
      if (localReplicationFactor > globalReplicationFactor) {
        throw error("Invalid local replication factor: {}, larger than global "
                    "replication factor: {}",
                    localReplicationFactor,
                    globalReplicationFactor);
      }
    }
  }
}

bool Ir::isCandidateForConstExprFolding(const Tensor &tensor) const {
  // A tensor is computable as a const expression if it is Const. This would
  // also be true for Variable tensors during inference, unless the user calls
  // resetHostWeights. Because of this, am choosing to ignore case of Variable
  // tensors during inference.
  auto tt = tensor.tensorType();
  return tt == TensorType::Const;
}

std::set<Tensor *> Ir::getRootInputsToOp(Op *op) {
  if (opAndRootInputs.find(op->id) != opAndRootInputs.end()) {
    // We have already stored the root inputs for this op
    // in a map. Retrieve here instead of performing search
    return opAndRootInputs.at(op->id);
  } else {
    std::set<Tensor *> rootInputs;

    // Get input tensors Ids
    std::vector<TensorId> inputIds = getTensors().getNoProducerIds();
    for (Tensor *tensor : op->input->tensors()) {
      if (std::find(inputIds.begin(), inputIds.end(), tensor->id) !=
          inputIds.end()) {
        // Tensor is a root input
        rootInputs.insert(tensor);
      } else {
        for (auto rootInputTensor : getRootInputsToOp(tensor->getProducer())) {
          rootInputs.insert(rootInputTensor);
        }
      }
    }

    // Add what we've found to the IR's map to speed up
    // future searches
    opAndRootInputs.emplace(op->id, rootInputs);

    return rootInputs;
  }
}

// Verify ConstExpr folding has removed input tensors that should have
// been removed:
//  - that initializer inputs are removed when possible in
//    inference mode
//  - that constant inputs are removed when possible in all modes
//
// 1. Get only the tensors we care about checking
// 2. For each tensor, get consumers
// 3. For each consumer, find its root input tensors
// 4. Confirm that at least on root input is not a candidate for
//    ConstExpr folding
//
// Note: this doesn't check that ConstExpr folding has removed
// tenosors that it shouldn't have
void Ir::verifyConstExprFolding() {
  for (auto id : getTensors().getNoProducerIds()) {
    Tensor *tensor = getTensors().get(id);

    // 1
    if (!isCandidateForConstExprFolding(*tensor)) {
      continue;
    }

    // 2 & 3
    std::set<Tensor *> rootInputs;
    for (auto consumingOp : tensor->consumers.getOps()) {
      for (auto rootInput : getRootInputsToOp(consumingOp)) {
        rootInputs.insert(rootInput);
      }
    }

    // 4
    bool shouldHaveFoldedTensor = true;
    for (auto rootInput : rootInputs) {
      if (!isCandidateForConstExprFolding(*rootInput)) {
        shouldHaveFoldedTensor = false;
      }
    }
    if (shouldHaveFoldedTensor) {
      logging::ir::warn(
          "ConstExpr folding has failed to remove input tensor {}, even though "
          "none of the root inputs to its consumers are variable tensors",
          tensor->id);
    }
  }
}

void Ir::prepare(const IrBundle &gb) {
  auto tryDumpIr = [&](auto logLevel) {
    auto irDumpDest = getPopartEnvVar("IR_DUMP");
    if (irDumpDest) {
      logging::log(logging::Module::ir,
                   logLevel,
                   logging::format("Writing ir to {}", irDumpDest));
      std::ofstream ofs;
      ofs.open(irDumpDest, std::ofstream::out);
      if (ofs.is_open()) {
        std::stringstream ss;
        serialise(Ir::SerialiseFormat::JSON, ss, false);
        ofs << ss.str();
      } else {
        logging::ir::err("Failed to open file {} to dump ir.", irDumpDest);
      }
    }
  };

  try {
    prepareImpl(gb);
  } catch (...) {
    tryDumpIr(logging::Level::Err);
    throw;
  }
  tryDumpIr(logging::Level::Debug);
}

void Ir::prepareImpl(const IrBundle &gb) {
  setDeviceInfo(gb.deviceInfo);

  if (isPrepared) {
    throw error("Ir::prepare called more than once");
  }

  if (gb.optimizer) {
    setExecutionMode(ExecutionMode::Training);
  } else {
    setExecutionMode(ExecutionMode::Inference);
  }

  setDataFlow(gb.dataFlow);
  setInputShapeInfo(gb.inputShapeInfo);
  setUserOptions(gb.userOptions);
  setPatterns(gb.patterns);
  setOnnxModel(gb.modelProto);

  if (graphs.size() == 1) {
    if (isPrepared) {
      throw error("There is more than one graph at the loss insertion stage, "
                  "which should not happen. This is an internal error.");
    }
  }

  if (canTrain()) {
    getMainGraph().setLoss(gb.loss);
  }

  confirmNoReservedIds();

  registerInputTensors();

  if (!canTrain() && getSessionOptions().enableGradientAccumulation) {
    throw error("Gradient Accumulation only available when training.");
  }

  logging::ir::info("Patterns : {}", patterns);
  // todo : validate the selected patterns

  // construct the forward pass from ONNX,
  constructForwards();
  if (!virtualGraphsEnabled()) {
    unsetAllVirtualGraphIds();
  }

  // Check virtual graph settings and annotations are consistent
  verifyVirtualGraphIds(false);
  verifyPipelineSettings();
  verifyPingPongSettings();
  verifyDistributedReplicatedGraphSettings();

  dotCheckpoint(DotCheck::Fwd0);

  for (auto &id_graph : graphs) {
    auto &graph = getGraph(id_graph.first);
    applyPreAliasPatterns(graph);
  }
  dotCheckpoint(DotCheck::Fwd1);

  if (requiresRandomSeed()) {
    initRandomSeed();
  }

  enableTransform(AutoVirtualGraph::id(),
                  userOptions.virtualGraphMode == VirtualGraphMode::Auto);
  applyTransform(AutoVirtualGraph::id(), getMainGraph());

  // Required transform order for PingPong is:
  // FWD -> PingPong1 -> BWD -> PingPong2 -> IpuCopy ->
  // PingPong3 -> Outline -> CacheSetup

  if (getSessionOptions().enablePipelining) {
    applyTransform(InferPipelineStages::id(), getMainGraph());
  }

  if (canTrain()) {
    setFinalLoss(gb.loss);
    updateVertices();
  }

  // First ping pong transformation pass (fwd)
  if (userOptions.virtualGraphMode == VirtualGraphMode::PingPong &&
      userOptions.pingPongPhases > 1) {
    applyTransform(PingPong::id(1), getMainGraph());
    verifyVirtualGraphIds(true);
  }

  // Batch serialisation, step 1
  // (has to occur before setNEdgesToLoss, but after setFinalLoss)
  if (userOptions.batchSerializationFactor > 1) {
    applyTransform(BatchSerialize::id(1), getMainGraph());
    updateVertices();
  }

  if (canTrain()) {
    setNEdgesToLoss();
  }

  if (autoRecomputationEnabled() && getMainGraph().hasUserRecomputeOps() &&
      getSessionOptions().pingPongPhases < 2) {
    throw error("A mixture of auto and manual recomputaion is not supported");
  }

  // tensors with no producer and no consumers are removed
  // at this point. We may want something more subtle.
  // (For pingpong the subtle thing here is to not remove cached tensors,
  // those special little snowflakes *)
  removeIsolatedTensors(true);

  if (gb.optimizer) {
    setOptimizer(*gb.optimizer);
  }

  updateVertices();
  if (canTrain()) {
    constructBackwards();
    verifyPipelineSettings();
  }

  updateVertices();
  dotCheckpoint(DotCheck::Bwd0);

  applyTransform(Prune::id(), getMainGraph());

  for (auto &id_graph : graphs) {
    auto &graph = getGraph(id_graph.first);
    applyPreAliasPatterns(graph);
  }

  if (canTrain()) {
    setNEdgesToLoss();
  }

  // tensors with no producer and no
  // consumers are removed at this point.
  removeIsolatedTensors(true);
  updateVertices();

  if (userOptions.explicitRecomputation) {
    logging::transform::warn(
        "WARNING: Enabling explicit recomputation. This option is deprecated "
        "and will be removed in a future version. Future versions will enable "
        "this option by default.");
    if (autoRecomputationEnabled() && getSessionOptions().pingPongPhases < 2) {
      logging::transform::info("Auto-annotating Ops for recomputation");
      recompute::autoAnnotate(getMainGraph(),
                              getSessionOptions().autoRecomputation);
    }
    // Transform from implicit to explicit recomputation
    applyTransform(ExplicitRecompute::id(), getMainGraph());
    updateVertices();
  }

  // Dynamicoptransform decomposes grad sums that contain
  // DynamicAdd/DynamicUpdate gradients, which can be decomposed efficiently
  applyTransform(DynamicOpTransform::id(), getMainGraph());

  // DecomposeGradSum decomposes remaining grad sums
  if (getSessionOptions().decomposeGradSum ||
      getSessionOptions().batchSerializationFactor > 1) {
    applyTransform(DecomposeGradSum::id(), getMainGraph());
  }

  switch (userOptions.mergeVarUpdate) {

  case (MergeVarUpdateType::All): {
    enableTransform(MergeAllVarUpdates::id(), true);
    applyTransform(MergeAllVarUpdates::id(), getMainGraph());
    updateVertices();
    break;
  }
  case (MergeVarUpdateType::AutoTight): {
    enableTransform(MergeTightThreshold::id(), true);
    applyTransform(MergeTightThreshold::id(), getMainGraph());
    updateVertices();
    break;
  }
  case (MergeVarUpdateType::AutoLoose): {
    enableTransform(MergeLooseThreshold::id(), true);
    applyTransform(MergeLooseThreshold::id(), getMainGraph());
    updateVertices();
    break;
  }

  case (MergeVarUpdateType::None): {
    // do nothing
    break;
  }

  case (MergeVarUpdateType::N):
  default: {
    // should never occur
    throw error("Unrecognised MergeVarUpdateType, bailing from merger");
  }
  }

  updateVertices();

  // we now start applying topological constraints between
  // Ops directly.
  if (canTrain()) {
    // 1. Ensure that the VarUpdate Ops are the final consumers
    //    of the Variable tensors
    getMainGraph().setVarUpdateConstraints();

    // 2. Ensure that ConvFlipWeights ops produce the transposed
    //    variable tensors only just before they are needed
    getMainGraph().setConvFlipWeightConstraints();
  }

  applyTransform(Prune::id(), getMainGraph());
  updateVertices();

  // Make sure that matmuls are serialized before gradient accumalation
  if (getSessionOptions().enableSerializedMatmuls) {
    applyTransform(SerializeMatMuls::id(), getMainGraph());
  }

  if (getSessionOptions().enableGroupedMatmuls) {
    applyTransform(GroupMatMuls::id(), getMainGraph());
  }

  // Accumulator Tensor for gradient accumulation / momentum is added here
  SGD1Decompose sgd1Decomposer;
  applyPreAliasPattern(&sgd1Decomposer, getMainGraph());

  if (getSessionOptions().hostWeightUpdate &&
      !getSessionOptions().hostAllReduce) {
    throw error(
        "Host weight update can't be enabled without enabling hostAllReduce.");
  }

  if (getSessionOptions().hostAllReduce) {
    if (canTrain()) {
      if (getSessionOptions().hostWeightUpdate &&
          !getSessionOptions().hostAllReduce) {
        throw error("Host weight update can't be enabled without enabling "
                    "hostAllReduce.");
      }
      if (userOptions.mergeVarUpdate != MergeVarUpdateType::None) {
        throw error("hostAllReduce does not work with MergeVarUpdates");
      }

      applyTransform(HostReduce::id(), getMainGraph());
      updateVertices();
    } else {
      logging::ir::info("Skipping hostAllReduce transform when running "
                        "inference or evaluation");
    }
  } else {
    if (getSessionOptions().enableReplicatedGraphs ||
        getSessionOptions().enableDistributedReplicatedGraphs) {
      applyTransform(DeviceReduce::id(), getMainGraph());
      updateVertices();
    }
  }

  // Add internal ops to copy tensors between ipu's as needed
  applyTransform(InterIpuCopy::id(), getMainGraph());

  // Pipelining optimizes copies separately, so only run if this is disabled
  if (!getSessionOptions().enablePipelining) {
    applyTransform(MergeCopies::id(), getMainGraph());
  }

  updateVertices();

  // Fourth ping pong transformation pass (cut)
  if (userOptions.virtualGraphMode == VirtualGraphMode::PingPong &&
      userOptions.pingPongPhases > 1) {
    // PingPong transformation 3 needs up-to-date aliasing information
    updateAliases();

    applyTransform(PingPong::id(2), getMainGraph());
    verifyVirtualGraphIds(true);
    // Remove extra CacheLoad, CacheStore and Replicated ops that are not used
    applyTransform(Prune::id(), getMainGraph());
  }

  updateVertices();

  // Batch serialisation, step 2
  if (userOptions.batchSerializationFactor > 1) {
    applyTransform(BatchSerialize::id(2), getMainGraph());
    updateVertices();
  }

  applyTransform(IoComputeTileCopy::id(), getMainGraph());
  updateVertices();

  for (auto &id_graph : graphs) {
    auto &graph = getGraph(id_graph.first);
    applyPreAliasPatterns(graph);
  }

  updateVertices();

  dotCheckpoint(DotCheck::PreAlias);

  if (getSessionOptions().enableOutlining) {
    updateAliases();
    applyTransform(SubgraphOutline::id(), getMainGraph());
    updateVertices();

    if (getSessionOptions().batchSerializationFactor > 1) {
      // Run a second outlining step.
      // This is necessary because in the first outlining pass we help the
      // outlining algorithm by inserting boundaries between
      // batch serialization phases.
      // Because batch serialization phases are not copied from the ops to their
      // parent subgraph, the second pass will ignore batch serialization phases
      // and outline the repeated per-batch-element subgraphs/ops.
      updateAliases();
      applyTransform(SubgraphOutline::id(), getMainGraph());
      updateVertices();
    }
  }

  removeIsolatedTensors(true);
  updateVertices();

  if (autoRecomputationEnabled() && !getSessionOptions().enablePipelining &&
      !getSessionOptions().explicitRecomputation &&
      getSessionOptions().pingPongPhases < 2) {
    updateVertices();
    logging::transform::info("Auto-annotating Ops for recomputation");
    recompute::autoAnnotate(getMainGraph(),
                            getSessionOptions().autoRecomputation);
  }

  updateVertices();

  // Each virtual graph is a pipeline stage in the pipeline.
  // Transform the graph to cache forward-pass tensors, and
  // restore them when needed in the backwards pass, allowing
  // for greater parallelism during compute.
  if (getSessionOptions().enablePipelining) {
    applyTransform(Pipeline::id(), getMainGraph());
    updateVertices();
  }

  applyTransform(MergeDuplicateOps::id(), getMainGraph());

  // Now, we apply the Patterns which can handle and create
  // topological constraints. Currently, this is only one
  // in-placing Pattern.
  if (patterns.isInPlaceEnabled()) {
    updateAliases();
    // Update the inplace priorities of ops before inplacing
    if (patterns.isUpdateInplacePrioritiesForIpuEnabled()) {
      applyUpdateInplacePrioritiesForIpu();
    }
    for (auto &id_graph : graphs) {
      applyInplacePattern(*id_graph.second);
    }
    updateVertices();
  }

  // Update aliases a final time
  updateAliases();

  applyTransform(CacheSetup::id(), getMainGraph());

  // confirm that all the anchor names provided
  // are indeed real tensor names. This is a check
  // that the user has not provided incorrect names.
  // We allow duplicates.
  validateAnchors();

  dotCheckpoint(DotCheck::Final);
  logIr();

  // some checks, now that prepare is complete
  for (auto &id_op : getMainGraph().getOps()) {
    if (id_op.second->opid == Onnx::CustomGradOperators::NllGrad) {
      logging::ir::warn("Computing gradient of the probabilities to Nll "
                        "might be less efficient than computing "
                        "pre-probability gradients directly with Pattern "
                        "SoftMaxGradDirect");
    }
  }

  verifyConstExprFolding();
  verifyConnectivity();
  verifyTensorIds();
  verifyVirtualGraphIds(true);
  verifyVertexAttributesOnlyInMain();
  verifySubgraphs();
  verifyRecomputeAttributes();
  // end of checks

  isPrepared = true;
}

void Ir::verifyVertexAttributesOnlyInMain() const {
  auto verify = [](Vertex *v) {
    if (v->toLoss != PathToLoss::Undefined) {
      throw error("Vertex {}, which is not in the main scope, does not have "
                  "PathToLoss::Undefined",
                  v->str());
    }
    if (v->fromLoss != PathFromLoss::Undefined) {
      throw error("Vertex {}, which is not in the main scope, does not have "
                  "PathFromLoss::Undefined",
                  v->str());
    }
    if (v->scheduledPreLoss != ScheduledPreLoss::Undefined) {
      throw error("Vertex {}, which is not in the main scope, does not have "
                  "ScheduledPreLoss::Undefined",
                  v->str());
    }
  };

  for (auto op : getOpSchedule({})) {

    // If this Vertex is not in the main Graph
    if (!op->settings.scope.str().empty()) {
      verify(op);

      for (auto tIn : op->input->tensors()) {
        verify(tIn);
      }

      for (auto tOut : op->output->tensors()) {
        verify(tOut);
      }
    }
  }
}

void Ir::verifyVirtualGraphIds(bool postAutoVirtualGraphTransform) const {
  if (!virtualGraphsEnabled()) {
    verifyVirualGraphIdsNotInitialized();
    return;
  }

  logging::ir::debug("Verifying virtual graph id consistency");

  const std::set<int64_t> &vgraphs(getMainGraph().getAllVirtualGraphIds());

  // a mix of annotated and not annotated Ops : suggests a problem
  if (vgraphs.count(Graph::NoVGraph) != 0 && vgraphs.size() > 1) {

    std::ostringstream errm;
    errm << "Either all Ops in the main graph must have their virtual "
         << "graph ids set, or none must. Op count per virtual graph id\n";

    std::map<int64_t, int> vgraph_op_count(
        getMainGraph().getVirtualGraphCounts());

    for (auto &id_size : vgraph_op_count) {
      errm << "  " << id_size.first << " : " << id_size.second << "\n";
    }

    errm << "Ops with no virtual graph id :  \n";
    for (auto &id_op : getMainGraph().getOps()) {
      auto op = id_op.second.get();
      if (!op->isConvertibleTo<IpuCopyOp>() && !op->hasVirtualGraphId()) {
        errm << "  " << op->str() << "\n";
      }
    }

    throw error(errm.str());
  }

  // Check number ipus makes sense given virtual graphs have been enabled
  if (!postAutoVirtualGraphTransform && deviceInfo->getNumIpus() == 1) {
    logging::ir::warn("Auto virtualGraphMode is on, but only one IPU is "
                      "specified, so no virtual graphs were created. Are you "
                      "sure you meant to set VirtualGraphMode to auto?");
  }

  // Sanity check the virtual graph ids. Only -1's, no Op has a virtual graph
  // annotation implies a problem.
  if (vgraphs.size() == 1 && vgraphs.count(-1) != 0) {
    // Manual virtual graphing, the user should have annotated ops.
    if (getSessionOptions().virtualGraphMode == VirtualGraphMode::Manual) {
      throw error("SessionOptions flag virtualGraphMode is {}, but no Ops "
                  "have been annotated with virtual graph information. This "
                  "is an inconsistent combination. ",
                  getSessionOptions().virtualGraphMode);
    }
    // Auto virtual graphing, why has the auto-sharder not run?
    else if (postAutoVirtualGraphTransform) {
      throw error("SessionOptions flag virtualGraphMode is {}, but no Ops have "
                  "been "
                  "annotated with virtual graph information. Moreover, the "
                  "paramater "
                  "postAutoVirtualGraphTransoform is true, so AutoVirtualGraph "
                  "should have been run. This is an inconsistent combination, "
                  "possibly an internal logic error has occured",
                  getSessionOptions().virtualGraphMode);
    }
  }
}

void Ir::verifyVirualGraphIdsNotInitialized() const {
  for (auto &id_graph : graphs) {
    auto &graph = id_graph.second;
    for (auto &id_op : graph->getOps()) {
      auto op = id_op.second.get();
      if (op->hasVirtualGraphId()) {
        std::ostringstream errm;
        errm << "SessionOptions flag virtualGraphMode is ";
        errm << getSessionOptions().virtualGraphMode;
        errm << ", but at least one op (";
        errm << op->debugName();
        errm << ") has virtualGraphId set.";

        throw error(errm.str());
      }
    }
  }
}

std::vector<TensorId> Ir::getModelInputIds() const {
  const auto &onnxGraph = onnxModel->graph();
  std::vector<TensorId> modelProtoInputIds;
  modelProtoInputIds.reserve(onnxGraph.input_size());
  for (const auto &valueInfo : onnxGraph.input()) {
    modelProtoInputIds.push_back(valueInfo.name());
  }
  return modelProtoInputIds;
}

void Ir::resetWeights(
    const ONNX_NAMESPACE::ModelProto &modelProto,
    const bool ignoreWeightsInModelWithoutCorrespondingIrWeight) {
  auto &onnxGraph = modelProto.graph();

  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    if (!getTensors().contains(tenId)) {
      if (ignoreWeightsInModelWithoutCorrespondingIrWeight) {
        continue;
      } else {
        throw error("resetWeights, no tensor '" + tenId + "' in tensors");
      }
    }
    auto tensor = getTensors().get(tenId);
    if (tensor->info != TensorInfo(initializer)) {
      throw error("trying to reset weights using tensor with non matching "
                  "tensor info. Tensor ID: {}",
                  tensor->id);
    }
    tensor->tensorData()->resetData(initializer);
  }
}

namespace {

void checkForDimParams(const TensorId &id, const ONNX_NAMESPACE::TypeProto &t) {
  auto dimString = [&]() {
    std::stringstream ss;
    ss << "[";
    int element_counter = 0;
    for (auto &v : t.tensor_type().shape().dim()) {
      if (element_counter > 0) {
        ss << ", ";
      }

      if (v.has_dim_param()) {
        ss << v.dim_param();
      } else {
        ss << v.dim_value();
      }
      element_counter += 1;
    }
    ss << "]";

    return ss.str();
  };

  for (auto &v : t.tensor_type().shape().dim()) {
    if (v.has_dim_param()) {
      throw error("Input tensor '{}' must be specified in InputShapeInfo, as "
                  "it has shape {}, which uses an unknown value '{}'.",
                  id,
                  dimString(),
                  v.dim_param());
    }
  }
}

} // namespace

void Ir::registerInputTensors() {

  auto &onnxGraph = onnxModel->graph();

  // Log the input tensor names, catch the
  // invalid case where they are repeated
  std::stringstream ss;
  std::set<TensorId> inputIds;
  bool repeatedInput = false;
  TensorId repeater  = "";
  ss << "Registering Input Tensors. ONNX Graph Inputs : [ ";
  for (auto &valueInfo : onnxGraph.input()) {
    TensorId id = valueInfo.name();
    ss << id << " ";
    if (inputIds.count(id) != 0) {
      // already seen, this is not valid. Will throw an error below.
      repeatedInput = true;
      repeater      = id;
    }
    inputIds.insert(id);
  }
  ss << "]";
  logging::debug(ss.str());
  if (repeatedInput) {
    throw error("Invalid ONNX Model : repeated name: ({}) in input list",
                repeater);
  }
  // we create a map of the tensors to their consumers' types
  std::map<TensorId, std::vector<std::string>> consumerTypes;
  auto addConsumerType = [&](const TensorId &tenId, const Node &node, int i) {
    auto found      = consumerTypes.find(tenId);
    auto consumerId = logging::format("{}@{}", node.op_type(), i);
    if (found == consumerTypes.end()) {
      consumerTypes[tenId] = {consumerId};
    } else {
      found->second.push_back(consumerId);
    }
  };

  // populate consumerTypes
  for (auto &node : onnxGraph.node()) {
    for (int i = 0; i < node.input_size(); ++i) {
      addConsumerType(node.input(i), node, i);
    }

    // need to look at the subgraph inputs for If, Call nodes
    auto addSubgraphInputs = [&](std::string branchName, Attributes attr) {
      auto branch = attr.getAttribute<Attributes::Graph>(branchName);
      for (int i = 0; i < branch.input_size(); i++) {
        auto inputId = branch.input(i).name();
        addConsumerType(inputId, node, i);
      }
    };
    if (node.op_type() == Onnx::AiOnnx::OpSet9::If.type) {
      Attributes attr{node.attribute()};
      addSubgraphInputs("then_branch", attr);
      addSubgraphInputs("else_branch", attr);
    }
    if (node.op_type() == Onnx::AiGraphcore::OpSet1::Call.type) {
      Attributes attr{node.attribute()};
      addSubgraphInputs("callee", attr);
    }
  }

  auto logCreationInfo = [&consumerTypes](std::string tensor_type,
                                          TensorId tensorId) {
    std::string consumerString = "";
    auto found                 = consumerTypes.find(tensorId);

    if (found == consumerTypes.end()) {
      consumerString = "with no consumers in the ONNX GraphProto";
    }

    else {
      consumerString = "with consumers [ ";
      for (auto &i : found->second) {
        consumerString += i;
        consumerString += " ";
      }
    }
    consumerString += "]";
    logging::info(
        "Adding {} Tensor {} to Ir {}.", tensor_type, tensorId, consumerString);
  };

  std::set<TensorId> onnxInitializers, unusedInitializers;

  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    if (consumerTypes.find(tenId) == consumerTypes.end()) {
      logging::info("Not creating Tensor for unused initializer, {}", tenId);
      unusedInitializers.emplace(tenId);
    } else {
      // If inference mode add initializers as constants if option enabled
      if (getExecutionMode() == ExecutionMode::Inference &&
          getSessionOptions().constantWeights == true) {
        logCreationInfo("Constant", tenId);
        getTensors().addConstInit(tenId, &initializer);
      } else {
        logCreationInfo("Variable", tenId);
        getTensors().addVarInit(tenId, &initializer);
      }
      onnxInitializers.emplace(tenId);
    }
  }

  // used onnx inputs which are not initializers are true inputs
  for (auto &valueInfo : onnxGraph.input()) {
    TensorId id = valueInfo.name();
    if (onnxInitializers.count(id) == 0 && unusedInitializers.count(id) == 0) {

      // Should we allow unused stream tensors in the ONNX Model? To be decided.
      bool allowUnusedStreamTensors = true;
      if (consumerTypes.find(id) == consumerTypes.end() &&
          !allowUnusedStreamTensors) {
        throw error("Request to create popart Stream Tensor {} failed, "
                    "as it has no consumers in the ONNX GraphProto. ",
                    id);
      }
      logCreationInfo("Stream", id);
      if (inputShapeInfo.has(id)) {
        getTensors().addStream(id, inputShapeInfo.get(id));
      } else if (valueInfo.has_type() &&
                 valueInfo.type().tensor_type().has_shape()) {
        checkForDimParams(id, valueInfo.type());
        getTensors().addStream(id, TensorInfo(valueInfo.type()));
      } else {
        throw error("Could not find tensor {} in InputShapeInfo, but no shape "
                    "is specified in the onnx model",
                    id);
      }

      // We will not be streaming data for this tensor from the host. Instead
      // initialise the tensor data once, here, based on the session option
      // syntheticDataMode
      if (useSyntheticData()) {
        Tensor *synStreamTensor = getTensor(id);
        const auto &info        = synStreamTensor->info;
        std::vector<char> data;
        std::vector<float> vals(info.nelms(), 0.0);

        switch (syntheticDataMode()) {
        case SyntheticDataMode::Zeros: {
          // Already initialized to zeros - do nothing
          break;
        }
        case SyntheticDataMode::RandomNormal: {
          // Random normal number generator: mean 0, variance 1
          // Boost Random ensures numerical consistency across implementations
          std::mt19937 generator;
          boost::random::normal_distribution<float> normalDistribution(0.0,
                                                                       1.0);
          for (auto &val : vals) {
            val = normalDistribution(generator);
          }
          break;
        }
        case SyntheticDataMode::Off:
        case SyntheticDataMode::N:
        default:
          throw error("Cannot set tensor data for current SyntheticDataMode");
        }

        for (float val : vals) {
          auto convertedData = convertFloatToDataType(info.dataType(), val);
          data.insert(data.end(), convertedData.begin(), convertedData.end());
        }
        synStreamTensor->setTensorData(info, data.data());
      }
    }
  }
}

void Ir::validateAnchors() const {
  for (TensorId id : dataFlow.anchors()) {
    if (!getTensors().contains(id)) {
      std::stringstream ss;
      ss << "Anchor tensor `" << id << "' not in Ir Tensors. ";
      // add some trouble-shooting for a case I stumbled upon:
      if (id.find(reservedGradientPrefix()) != std::string::npos) {
        std::string degrad = getNonGradId(id);
        if (getTensors().contains(degrad)) {
          ss << "\nInterestingly, `" << degrad << '\'' << " IS in tensors.\n";
          ss << "Note that not all tensors can have their gradients "
             << "anchored:\nif an activation tensor does not lead "
             << "to the loss,\nits gradient is zero and never computed.";
        }
      } else {
        ss << "The tensors are:\n";
        getTensors().append(ss);
      }
      throw error(ss.str());
    }
  }
}

bool Ir::applyPreAliasPattern(const PreAliasPattern *pattern, Graph &graph) {
  bool result = false;

  auto touchesInputToLoss = [&graph, pattern](Op *op) {
    for (auto &tensor : pattern->touches(op)) {
      if (graph.getTensors().contains(graph.getLoss())) {
        if (graph.getLoss() == tensor->id) {
          return true;
        }
      }
    }
    return false;
  };

  auto canApplyPattern = [this, &touchesInputToLoss, pattern](Op *op) {
    if (op->isExcludedFromPattern(pattern) || !pattern->matches(op) ||
        pattern->touchesAnchored(op)) {
      return false;
    }

    // If the ir will construct a loss, but hasn't yet, check that the pattern
    // doesn't touch the inputs to the loss.
    if (this->canTrain() && !this->constructedFinalLoss &&
        touchesInputToLoss(op)) {
      return false;
    }

    return true;
  };

  // the pattern chooses what order to go through the ops in

  std::vector<OpId> v_ops;
  v_ops.reserve(graph.getOps().size());

  for (auto &id_op : graph.getOps()) {
    v_ops.push_back(id_op.first);
  }

  for (auto opId : v_ops) {
    auto itr = graph.getOps().find(opId);

    // If the op still exists
    if (itr != graph.getOps().end()) {
      Op *op = itr->second.get();
      if (canApplyPattern(op)) {
        logging::pattern::debug("Applying pattern {} to {}",
                                pattern->getPatternName(),
                                op->debugName());
        result |= pattern->apply(op);
      }
    }
  }

  return result;
}

void Ir::applyPreAliasPatterns(Graph &graph) {

  bool keepRunning = true;
  std::vector<std::unique_ptr<PreAliasPattern>> pList =
      patterns.getPreAliasList();

  while (keepRunning) {
    foldConstants(graph);

    keepRunning = false;
    for (auto &pattern : pList) {
      keepRunning |= applyPreAliasPattern(pattern.get(), graph);
    }
  }
}

void Ir::applyTransform(std::size_t transformId, Graph &graph) {
  // Unless explictly set, a transform is enabled
  if (transformEnableMap.count(transformId) == 0 ||
      transformEnableMap.at(transformId)) {
    Transform::applyTransform(transformId, graph);
  }
}

void Ir::enableTransform(std::size_t transformId, bool enable) {
  transformEnableMap[transformId] = enable;
}

std::vector<Op *> Ir::opsOfType(const OperatorIdentifier &opid) {
  std::vector<Op *> typedOps;
  for (auto &id_graph : graphs) {
    auto graph = id_graph.second.get();

    for (auto &id_op : graph->getOps()) {
      if (id_op.second->opid == opid) {
        typedOps.push_back(id_op.second.get());
      }
    }
  }
  return typedOps;
}

bool Ir::isConsumedByOpOfType(TensorId tid, const OperatorIdentifier &opid) {
  auto tensor       = getTensors().get(tid);
  auto tidConsumers = tensor->consumers.getOps();

  for (Op *op : tidConsumers) {
    if (op->opid == opid) {
      return true;
    }
  }
  return false;
}

bool Ir::isAnchored(const TensorId &tenId) const {
  return dataFlow.isAnchored(tenId);
}

bool Ir::streamingIsDisabledForTensor(const TensorId &tensorId) const {
  // What conditions mean that this tensor should not be streamed?

  // 1. Streams have been turned off globally
  if (useSyntheticData()) {
    return true;
  }

  // 2. The tensor is an Gradient Accl tensor, but the user
  //    has turned off streaming for this kind of tensor
  if (getTensors().get(tensorId)->isAcclTensor() &&
      getSessionOptions().disableGradAccumulationTensorStreams) {
    return true;
  }

  // 3. The tensor is cached
  if (getTensors().get(tensorId)->cacheInfo.isCached()) {
    return true;
  }

  return false;
}

bool Ir::storingIsDisabledForTensor(const TensorId &tensorId) const {
  // What conditions mean that this tensor should not be streamed?

  // 1. Streams have been turned off globally
  if (useSyntheticData()) {
    return true;
  }

  // 2. The tensor is an Gradient Accl tensor, but the user
  //    has turned off streaming for this kind of tensor
  if (getTensors().get(tensorId)->isAcclTensor() &&
      getSessionOptions().disableGradAccumulationTensorStreams) {
    return true;
  }

  return false;
}

void Ir::constructForwards() {
  constructFromOnnxGraph(onnxModel->graph(), {});
  for (auto &id_op : getMainGraph().getOps()) {
    auto op      = id_op.second.get();
    op->fromLoss = PathFromLoss::No;
  }
}

Graph &Ir::constructFromOnnxGraph(const ONNX_NAMESPACE::GraphProto &graph,
                                  const Scope &scope) {
  auto scope_id = scope.str();
  if (graphs.find(scope_id) == graphs.end()) {
    logging::ir::debug("Adding new graph for scope {}", scope_id);
    graphs.insert({scope_id, std::make_unique<Graph>(*this, scope_id)});
  }

  graphs.at(scope_id)->constructFromOnnxGraph(graph);

  return *graphs.at(scope_id);
}

void Ir::foldConstants(Graph &graph) {
  logging::ces::trace("Folding constants");
  ConstExprUtil::foldConstants(graph);
}

OpId Ir::getAndIncrOpsCounter() {
  OpId nOps0 = opsCounter;
  ++opsCounter;
  return nOps0;
}

OpId Ir::getOpsCounter() const { return opsCounter; }

OptionalVGraphId
Ir::getVirtualGraphIdFromTensorProducers(std::vector<Tensor *> ts) {
  // Count which vgraph's the producer ops are on.
  std::map<int64_t, int64_t> vgraphIdMap;
  for (auto &t : ts) {
    Op *producer = t->getProducerUnsafe();
    if (producer) {
      if (producer->hasVirtualGraphId()) {
        vgraphIdMap[producer->getVirtualGraphId()]++;
      }
    }
  }

  if (vgraphIdMap.size() == 0) {
    std::vector<TensorId> ts_ids;
    for (auto t : ts) {
      ts_ids.push_back(t->id);
    }
    throw internal_error(
        "None of the producers of the tensors in {} have virtual "
        "graph ids",
        ts_ids);
  }

  // Find the vgraph id with the most occurrences.
  auto it = std::max_element(vgraphIdMap.begin(),
                             vgraphIdMap.end(),
                             [](const std::pair<int64_t, int64_t> &p1,
                                const std::pair<int64_t, int64_t> &p2) {
                               return p1.second < p2.second;
                             });

  return OptionalVGraphId(it->first);
}

std::string Ir::getGradSumOpNamePrefix() const { return "GradSum"; }

Op *Ir::growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum) {

  std::unique_ptr<popart::Op> gradSum =
      OpManager::createOp(Domain::ai_onnx,
                          "Sum",
                          getOpSetVersionFromModel(Domain::ai_onnx),
                          getMainGraph(),
                          getGradSumOpNamePrefix());

  OpId opId = getMainGraph().moveIntoGraph(std::move(gradSum));

  std::vector<TensorId> inputs;
  inputs.reserve(toSum.size());
  for (auto &tensor : toSum) {
    inputs.push_back(tensor->id);
  }
  TensorId gradientId = getGradId(target->id);
  std::vector<TensorId> outputs{gradientId};

  getMainGraph().connectInputs(InputVecWrapper(inputs), opId);
  getMainGraph().connectOutputs(OutputVecWrapper(outputs), opId);
  Op *op = getMainGraph().getOps()[opId].get();
  op->setup();
  op->inheritPlacementAttributes(true);
  return op;
}

PipelineStage Ir::getFinalLossPipelineStage() const {
  auto finalLossOpFound = getMainGraph().getOps().find(finalLossOpId);
  if (finalLossOpFound != getMainGraph().getOps().end()) {
    auto lossOp = finalLossOpFound->second.get();
    return lossOp->getPipelineStage();
  } else {
    throw error("Could not find final loss to get PipelineStage from");
  }
}

int64_t Ir::getNumPipelineStages() const {
  std::set<PipelineStage> pStages;

  for (auto &id_op : getMainGraph().getOps()) {
    auto op = id_op.second.get();
    if (op->hasPipelineStage()) {
      pStages.insert(op->getPipelineStage());
    }
  }
  int64_t numStages = pStages.size();

  // Check there are no 'missing' pipeline stages
  for (int64_t i = 0; i < numStages; i++) {
    if (!pStages.count(i)) {
      throw error("The set of pipeline stages for all Ops contains {} stages, "
                  "but stage {} is missing",
                  numStages,
                  i);
    }
  }
  return numStages;
}

std::vector<Op *> Ir::growGradOps(Op *nonGradOp) {
  PipelineStage maxPipelineStage = 0;
  if (getSessionOptions().enablePipelining) {
    // the last fwd pass pipeline stage is also the first bwd pass pipeline
    // stage.
    maxPipelineStage = getFinalLossPipelineStage() * 2;
  }

  OpId nonGradOpId = nonGradOp->id;
  auto backOps     = nonGradOp->getGradOps();
  if (backOps.size() < 1) {
    logging::ir::debug("Cannot get gradients for {}", nonGradOp->debugName());
  }
  std::vector<Op *> gradOps;
  for (auto &upop : backOps) {
    Op *gradOp    = upop.get();
    OpId gradOpId = getMainGraph().moveIntoGraph(std::move(upop));

    // Reset priority, since fwd priority should not influence bwd priority
    //
    // TODO: Uncomment this. This prevented explicit priorities on certain
    // gradient ops being set which was necessary as a short term fix for
    // sharded training regressions seen in T17036. This could be replaced
    // once explicit priorities are no longer needed for this purpose. T17311
    // should fix this.
    //
    // gradOp->settings.schedulePriority = 0.0;

    if (gradOp->hasPingPongPhase()) {
      // Remap from forward to backward pingpong phase
      gradOp->setPingPongPhase(2 * getSessionOptions().pingPongPhases - 2 -
                               gradOp->getPingPongPhase());
    }

    if (nonGradOp->settings.recomputeType == RecomputeType::Recompute &&
        autoRecomputationEnabled() && getSessionOptions().pingPongPhases < 2) {
      throw error("Grad Ops should be grown before recompute annotation");
    }

    // No gradOp should be of type Recompute.
    gradOp->settings.recomputeType = RecomputeType::Checkpoint;

    if (nonGradOp->hasPipelineStage()) {
      gradOp->setPipelineStage(maxPipelineStage -
                               nonGradOp->getPipelineStage());
    }

    // connect inputs of gradOp
    {
      // inputs to gradOp (to populate in this scope):
      std::map<int, std::string> m_inputs;
      auto isInputOptional = [](Op *op, InIndex i) {
        auto optionalInputs = op->optionalInputs();
        return optionalInputs.find(i) != optionalInputs.end();
      };
      for (auto &inOutMapper : gradOp->gradInputInfo()) {

        int indexGrad     = inOutMapper.iGrad;
        int indexFwd      = inOutMapper.iNonGrad;
        GradOpInType type = inOutMapper.type;

        // the input at index 'indexGrad' to gradOp is
        switch (type) {
        //  (1) the INPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::In: {
          if (nonGradOp->input->hasIndex(indexFwd)) {
            m_inputs[indexGrad] = nonGradOp->input->tensor(indexFwd)->id;
          } else if (isInputOptional(nonGradOp, indexFwd)) {
            m_inputs[indexGrad] = TensorId();
          } else {
            throw error(
                "Invalid configuration of gradOp {}. nonGradOp ({}) INPUT {} "
                "is not marked as optional, but is not defined",
                gradOp->debugName(),
                nonGradOp->debugName(),
                indexFwd);
          }
          break;
        }

        //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::Out: {
          if (!nonGradOp->output->hasIndex(indexFwd)) {
            throw error("Invalid configuration of gradOp {}. nonGradOp ({}) "
                        "OUTPUT {} is not defined ",
                        gradOp->debugName(),
                        nonGradOp->debugName(),
                        indexFwd);
          }
          m_inputs[indexGrad] = nonGradOp->output->tensor(indexFwd)->id;
          break;
        }

        //  (3) the GRADIENT of the OUTPUT
        //      at index 'indexFwd' of nonGradOp.
        case GradOpInType::GradOut: {
          if (!nonGradOp->output->hasIndex(indexFwd)) {
            throw error("Invalid configuration of gradOp {}. nonGradOp ({}) "
                        "OUTPUT {} is not defined ",
                        gradOp->debugName(),
                        nonGradOp->debugName(),
                        indexFwd);
          }

          auto gradTensorId =
              getGradId(nonGradOp->output->tensor(indexFwd)->id);
          if (getMainGraph().getTensors().contains(gradTensorId,
                                                   gradOp->getScope())) {
            m_inputs[indexGrad] = gradTensorId;
          } else {
            if (isInputOptional(gradOp, indexGrad)) {
              m_inputs[indexGrad] = TensorId();
            } else {
              throw error("No gradient for non-grad-op {} at index {}, but "
                          "input {} is not marked as optional on grad-op {}. "
                          "Could it be that "
                          "the path along that index did not lead to the final "
                          "loss, in which case the gradient is zero?",
                          nonGradOp->debugName(),
                          indexFwd,
                          indexGrad,
                          gradOp->debugName());
            }
          }
          break;
        }
        }
      }

      getMainGraph().connectInputs(InputMapWrapper(m_inputs), gradOpId);
    }

    // connect outputs of gradOp
    {
      std::vector<TensorId> v_outputs;
      for (auto out_in : gradOp->gradOutToNonGradIn()) {
        int gradOut   = out_in.first;
        int nonGradIn = out_in.second;

        if (v_outputs.size() < gradOut + 1) {
          v_outputs.resize(gradOut + 1, TensorId());
        }

        if (nonGradOp->input->hasIndex(nonGradIn)) {
          TensorId inId      = nonGradOp->input->tensor(nonGradIn)->id;
          TensorId outId     = getEdgeGradId(inId, nonGradOpId, nonGradIn);
          v_outputs[gradOut] = outId;
        }
      }
      getMainGraph().connectOutputs(OutputVecWrapper(v_outputs), gradOpId);
    }
    gradOp->setup();

    // note, as the outputs of gradOp are edge-grad-tensors and not
    // edge-grads, we do not need to match them to non-grad tensors.
    gradOps.push_back(gradOp);
  }

  return gradOps;
}

void TensorGradRegistry::insert(Tensor *nonGrad, Tensor *grad) {
  // The expected number of edges is assumed to be the same as the
  // number of edges to the loss for the non-grad tensor.
  if (expectedNumEdges.find(nonGrad->id) == expectedNumEdges.end()) {
    expectedNumEdges.insert({nonGrad->id, nonGrad->nEdgesToLoss});
  }

  auto found = partial.find(nonGrad->id);
  if (found == partial.end()) {
    partial.insert({nonGrad->id, {grad}});
  } else {
    partial[nonGrad->id].push_back(grad);
  }

  tryMakeComplete(nonGrad);
}

void TensorGradRegistry::decrementNumberExpectedEdges(Tensor *nonGrad) {
  auto found = expectedNumEdges.find(nonGrad->id);
  if (found == expectedNumEdges.end()) {
    expectedNumEdges.insert({nonGrad->id, nonGrad->nEdgesToLoss - 1});
  } else {
    found->second--;
  }

  // Only make complete if this is already in partials.
  // This prevents adding entries with 0 gradient edges.
  if (partial.find(nonGrad->id) != partial.end()) {
    tryMakeComplete(nonGrad);
  }
}

int TensorGradRegistry::getNumberExpectedEdges(Tensor *nonGrad) {
  auto found = expectedNumEdges.find(nonGrad->id);
  if (found != expectedNumEdges.end()) {
    return found->second;
  } else {
    return nonGrad->nEdgesToLoss;
  }
}

void TensorGradRegistry::tryMakeComplete(Tensor *nonGrad) {
  if (partial[nonGrad->id].size() == expectedNumEdges.at(nonGrad->id)) {
    complete[nonGrad->id] = partial[nonGrad->id];
    partial.erase(nonGrad->id);
  }
}

void OpGradRegistry::insert(Op *nonGrad, int index) {
  auto found = partial.find(nonGrad->id);
  // so far NO gradients for nonGrad are in:
  if (found == partial.end()) {
    partial.insert({nonGrad->id, {}});
  }
  // this should be removed when we're happy the IL (internal logic)
  // is correct:
  if (partial[nonGrad->id].count(index) != 0) {
    throw internal_error("index already present in OpGradRegistry::insert");
  }

  partial[nonGrad->id].insert(index);

  // probably just checks that the size of partial is
  // nonGrad->output->n(), but maybe not.
  if (nonGrad->readyToCreateGradients(partial[nonGrad->id])) {
    complete.push_back(nonGrad);
    partial.erase(nonGrad->id);
  }
}

std::map<TensorId, std::vector<Tensor *>> TensorGradRegistry::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

std::vector<Op *> OpGradRegistry::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

// design choice: we could have an "irHasChanged"
// flag which is set to true whenever the Ir changes,
// and then if irHasChanged is false, calls
// to this (and other) functions can do nothing.
// The cost of maintaining irHasChanged is non-trivial
// and would require runtime overhead, for now not
// going to implement it.
//

namespace {

// move backwards through the inputs and their producers
std::set<Vertex *> backwardPropogate(std::vector<Op *> frontier) {
  std::set<Vertex *> visited;
  for (auto x : frontier) {
    visited.emplace(x);
  }
  while (frontier.size() > 0) {
    auto toProcess = frontier.back();
    frontier.resize(frontier.size() - 1);
    // get all producers of inputs, add them to the frontier
    for (auto inTensor : toProcess->input->tensors()) {
      visited.emplace(inTensor);
      auto producer = inTensor->getProducerUnsafe();
      if (producer && visited.count(producer) == 0) {
        visited.emplace(producer);
        frontier.push_back(producer);
      }
    }
  }
  return visited;
}

// move forwards the the outputs and their consumers
std::set<Vertex *> forwardPropogate(std::vector<Op *> frontier) {
  std::set<Vertex *> visited;
  for (auto x : frontier) {
    visited.emplace(x);
  }
  while (frontier.size() > 0) {
    auto toProcess = frontier.back();
    frontier.resize(frontier.size() - 1);
    for (auto outTensor : toProcess->output->tensors()) {
      visited.emplace(outTensor);
      for (auto consumer : outTensor->consumers.getOps()) {
        if (visited.count(consumer) == 0) {
          visited.emplace(consumer);
          frontier.push_back(consumer);
        }
      }
    }
  }
  return visited;
}

} // namespace

void Ir::updateVertices() {

  // for all vertices (Ops and Tensors), set
  //  1) toLoss (is there a path to the final loss?)
  //  2) fromLoss (is there a path from the final loss?)
  //  3) scheduledPreLoss (is it scheduled before the final loss?)

  logging::ir::trace(
      "\nUpdating all Vertices (toLoss, fromLoss, scheduledPreLoss)");

  // 1) Get all Ops which have toLoss Yes, and backwards propagate
  std::vector<Op *> toLossFrontier;
  // 2) Get all Ops which have fromLoss Yes, and forwards propagate
  std::vector<Op *> fromLossFrontier;
  for (auto &id_op : getMainGraph().getOps()) {
    Op *op = id_op.second.get();
    if (op->toLoss == PathToLoss::Yes) {
      toLossFrontier.push_back(op);
    }

    if (op->fromLoss == PathFromLoss::Yes) {
      fromLossFrontier.push_back(op);
    }

    // If an Op's input has PathFromLoss::Yes, then so do does Op
    for (auto arr : op->input->tensors()) {
      if (arr->fromLoss == PathFromLoss::Yes) {
        op->fromLoss = PathFromLoss::Yes;
        fromLossFrontier.push_back(op);
      }
    }

    // If an Op's output has PathToLoss::Yes, then so do does Op
    for (auto arr : op->output->tensors()) {
      if (arr->toLoss == PathToLoss::Yes) {
        op->toLoss = PathToLoss::Yes;
        toLossFrontier.push_back(op);
      }
    }
  }

  auto toLossVertices = backwardPropogate(toLossFrontier);
  for (Vertex *v : toLossVertices) {
    v->toLoss = PathToLoss::Yes;
  }

  auto fromLossVertices = forwardPropogate(fromLossFrontier);
  for (Vertex *v : fromLossVertices) {
    v->fromLoss = PathFromLoss::Yes;
  }

  // set all Undefined to No
  for (auto &id_op : getMainGraph().getOps()) {
    auto setUnPaths = [](Vertex *v) {
      if (v->toLoss == PathToLoss::Undefined) {
        v->toLoss = PathToLoss::No;
      }
      if (v->fromLoss == PathFromLoss::Undefined) {
        v->fromLoss = PathFromLoss::No;
      }
    };

    auto op = id_op.second.get();
    setUnPaths(op);
    for (auto tensor : op->input->tensors()) {
      setUnPaths(tensor);
    }
    for (auto tensor : op->output->tensors()) {
      setUnPaths(tensor);
    }
  }

  // 3.1) scheduledPreLoss for Ops.
  // Op which have PathFromLoss::Yes are ScheduledPreLoss::No
  for (auto op : getMainGraph().getOpSchedule({})) {
    if (op->fromLoss == PathFromLoss::Yes) {
      op->scheduledPreLoss = ScheduledPreLoss::No;
    } else {
      op->scheduledPreLoss = ScheduledPreLoss::Yes;
    }
    if (op->scheduledPreLoss == ScheduledPreLoss::No &&
        op->settings.recomputeType != RecomputeType::Recomputed) {
      op->settings.recomputeType = RecomputeType::Checkpoint;
    }
  }

  logging::ir::debug("setting scheduledPreLoss for Tensors in updateVertices");
  // 3.2) scheduledPreLoss for Tensors and any ops occuring post the loss
  // in the schedule
  bool postLoss = false;
  for (auto op : getMainGraph().getOpSchedule({})) {
    postLoss |= op->scheduledPreLoss == ScheduledPreLoss::No;
    if (postLoss) {
      // The loss has been crossed, everything ScheduledPreLoss::No from here on
      op->scheduledPreLoss = ScheduledPreLoss::No;
    }
    for (auto tensor : op->input->tensors()) {
      // inputs to pre-loss are pre-loss
      if (op->scheduledPreLoss == ScheduledPreLoss::Yes) {
        tensor->scheduledPreLoss = ScheduledPreLoss::Yes;
        // inputs to post-loss are post-loss if not already pre-loss
      } else if (op->scheduledPreLoss == ScheduledPreLoss::No) {
        if (tensor->scheduledPreLoss != ScheduledPreLoss::Yes) {
          tensor->scheduledPreLoss = ScheduledPreLoss::No;
        }
      }
    }
    // Outputs are always the same as the producer Op, this rule takes priority
    // over all input annotation rules.
    for (auto tensor : op->output->tensors()) {
      tensor->scheduledPreLoss = op->scheduledPreLoss;
    }
  }
}

void Ir::updateAliases() {
  for (auto &graph : graphs) {
    graph.second->getTensors().clearAliases();
    for (auto &op : graph.second->getOps()) {
      graph.second->getTensors().updateAliases(op.second.get());
    }
  }
}

void Ir::unsetAllVirtualGraphIds() {
  bool hadToUnsetAny = false;

  for (auto &id_graph : graphs) {
    auto &graph = id_graph.second;
    for (auto &id_op : graph->getOps()) {
      auto op = id_op.second.get();

      if (op->hasVirtualGraphId()) {
        // no virtual graph id
        op->setVirtualGraphId({});
        hadToUnsetAny = true;
      }
    }
  }

  if (hadToUnsetAny) {
    logging::ir::warn("Virtual graph settings ignored because virtual "
                      "graphs are not enabled.");
  }
}

void Ir::setNEdgesToLoss() {

  if (isTesting()) {
    throw internal_error(
        "Call to setNEdgesToLoss() in Testing  mode is not valid");
  }

  // set all edge counts to zero (we set from scratch in this function)
  for (auto &id_op : getMainGraph().getOps()) {
    Op *op           = id_op.second.get();
    op->nEdgesToLoss = 0;
    for (auto index_tensor : op->input->tensorMap()) {
      index_tensor.second->nEdgesToLoss = 0;
    }
    for (auto index_tensor : op->output->tensorMap()) {
      index_tensor.second->nEdgesToLoss = 0;
    }
  }

  for (auto &id_op : getMainGraph().getOps()) {
    Op *op = id_op.second.get();

    // For each Op, how many OutIndices lead to loss?
    for (auto index_tensor : op->output->tensorMap()) {
      auto outTensor = index_tensor.second;
      if (outTensor->toLoss == PathToLoss::Yes) {
        ++op->nEdgesToLoss;
      }
    }

    // If Op goes to Loss, then for each of its inputs, +1 path
    if (op->toLoss == PathToLoss::Yes) {
      for (auto index_tensor : op->input->tensorMap()) {
        auto inTensor = index_tensor.second;
        ++inTensor->nEdgesToLoss;
      }
    }
  }
}

void Ir::constructBackwards() {

  logging::ir::info("Constructing backwards pass");

  // definition: edge-gradient. What is output by a grad-op,
  // and which will be summed with other edge-gradients to create
  // a gradient. It is possible that an edge-gradient has the same
  // value as a gradient, if a tensor has only 1 consumer.

  // design decision w.r.t. lambda functions in this function:
  // see-sawing between lambda functions (see two following here)
  // and member functions. In general I don't like lambda functions,
  // their return types are not easily visible and capturing parameters
  // is tedious. However, I also don't like having class variables
  // which are only used in one bit of functionality, because it becomes
  // unclear whether they should be maintained in a valid state throughout
  // the objects life. In this case, I think the second is worse, so
  // going for the lambda solution.

  TensorGradRegistry tensor_grad_registry;
  OpGradRegistry op_grad_registry;

  // signal that a grad-op has created edge-gradients
  auto registerOpGrads = [&tensor_grad_registry](Op *gradOp, Op *nonGradOp) {
    for (auto &index_tensor : gradOp->output->tensorMap()) {
      int opOutInd     = index_tensor.first;
      Tensor *partGrad = index_tensor.second;
      // what input index of nonGradOp does the
      // edge-gradient correspond to?
      int nonGradInInd      = gradOp->getNonGradInIndex(opOutInd);
      Tensor *nonGradTensor = nonGradOp->input->tensor(nonGradInInd);
      tensor_grad_registry.insert(nonGradTensor, partGrad);
    }
  };

  // register an op that doesn't create any grad ops
  std::function<void(Op *)> registerOpWithoutGrads;
  registerOpWithoutGrads = [&tensor_grad_registry,
                            &registerOpWithoutGrads](Op *nonGradOp) {
    for (auto &index_tensor : nonGradOp->input->tensorMap()) {
      auto input = index_tensor.second;
      tensor_grad_registry.decrementNumberExpectedEdges(input);

      if (tensor_grad_registry.getNumberExpectedEdges(input) == 0 &&
          input->hasProducer()) {
        registerOpWithoutGrads(input->getProducer());
      }
    }
  };

  // communicate that a new gradient tensor
  // (which is a sum along edges) is ready
  auto registerTensorGrad = [this, &op_grad_registry](Tensor *sum) {
    Tensor *nonGrad = getTensors().get(getNonGradId(sum->id));
    if (nonGrad->hasProducer()) {
      Op *producer = nonGrad->getProducer();
      // the index at which nonGrad was produced
      int index = producer->output->indices(nonGrad).at(0);
      op_grad_registry.insert(producer, index);
    }
  };

  // grad-ops which have created edge-gradients, but the
  // edge-gradients haven't signalled their existance.
  // initialised as the gradients of the loss
  std::vector<GradNonGradPair> opsToRegister = growLossGradients();

  while (!opsToRegister.empty() || !tensor_grad_registry.complete.empty()) {

    if (!opsToRegister.empty()) {
      auto &toRegister = opsToRegister.back();
      registerOpGrads(toRegister.grad, toRegister.nongrad);
      opsToRegister.resize(opsToRegister.size() - 1);
    }

    for (auto &nongrad_egrads : tensor_grad_registry.popComplete()) {
      Tensor *nongrad = getTensors().get(nongrad_egrads.first);

      const std::vector<Tensor *> &egrads = nongrad_egrads.second;
      // nongrad required below, as the name of the output of the
      // created op (sumOp) will be based off of it. Also, we
      // register the link between sumOp's output and nongrad
      Op *sumOp       = growGradSumOp(nongrad, egrads);
      sumOp->fromLoss = PathFromLoss::Yes;

      switch (nongrad->tensorType()) {
      // if sumOp creates the gradient of an activation tensor,
      case TensorType::ActGrad: {
        registerTensorGrad(sumOp->output->tensor(0));
        break;
      }
      case TensorType::Variable: {
        // nothing to do, variable updates
        // follows at the end of this function
        break;
      }
      case TensorType::Stream: {
        // if the user wants the gradient of the
        // input data (unusual case) maybe we won't
        // break here. Example case : generating adversarials
        break;
      }
      case TensorType::Const: {
        break;
      }
      case TensorType::Cache:
      case TensorType::Momentum:
      case TensorType::Unknown:
      case TensorType::N:
        throw error("can't currently register gradient of " +
                    nongrad->tensor_type() + " tensor, " + nongrad->str());

      default:
        throw error("only handling ActGrad and Variable for now");
      }
    }

    for (Op *op : op_grad_registry.popComplete()) {
      auto gradOps = growGradOps(op);
      if (gradOps.size() == 0) {
        registerOpWithoutGrads(op);
      } else {
        for (auto &gradOp : gradOps) {
          opsToRegister.push_back({gradOp, op});
        }
      }
    }
  }

  logging::ir::info("Creating Variable Tensor update Ops");
  // add weight update ops (we are ignoring momentums for now)
  for (auto &varId : getTensors().getIds(TensorType::Variable)) {

    VariableTensor *tensor =
        dynamic_cast<VariableTensor *>(getTensors().get(varId));
    switch (tensor->getVariableUpdateType()) {
    case VariableUpdateType::Copy:
      // Updates the var by copying it from another tensor
      growCopyVarUpdateOp(varId, tensor->getCopyFromTensor());
      break;
    case VariableUpdateType::Gradient:
      // Updates the var by looking for the matching gradient
      growGradientVarUpdateOp(varId);
      break;
    case VariableUpdateType::None:
    default:
      throw error("Unknown variable update approach");
    }
  }

  // All Ops and Tensors at this point with a reserved gradient prefix have a
  // path from the final Loss (before any Patterns and Transformations). After
  // Patterns, this is no longer true as names get mangled.
  for (auto &id_op : getMainGraph().getOps()) {
    Op *op = id_op.second.get();
    for (auto inArr : op->input->tensors()) {
      if (inArr->id.find(reservedGradientPrefix()) != std::string::npos) {
        inArr->fromLoss = PathFromLoss::Yes;
        op->fromLoss    = PathFromLoss::Yes;
      }
    }
    for (auto outArr : op->output->tensors()) {
      if (outArr->id.find(reservedGradientPrefix()) != std::string::npos) {
        outArr->fromLoss = PathFromLoss::Yes;
        op->fromLoss     = PathFromLoss::Yes;
      }
    }
  }
  logging::ir::info("Constructing backwards complete");
  constructedBackwards = true;
}

void Ir::growCopyVarUpdateOp(const TensorId &varId, const TensorId &from) {
  OpId opId = getMainGraph().moveIntoGraph(
      std::unique_ptr<Op>(new CopyVarUpdateOp(varId, {getMainGraph(), ""})));

  // The order of inputs is important
  std::vector<TensorId> inputs{varId, from};
  getMainGraph().connectInputs(InputVecWrapper(inputs), opId);

  growVarUpdateOpInternal(opId);
}

void Ir::growGradientVarUpdateOp(const TensorId &varId) {

  logging::ir::info("Growing gradient var update op for {}", varId);

  // A sanity check that the Tensor is not fixed point type
  if (getTensors().get(varId)->info.getDataTypeInfo()->isFixedPoint()) {
    throw error("Currently only floating point variable tensors are updatable");
  }

  const Tensor &var = *getTensors().get(varId);
  auto inputIds     = optimizer->getInputIds(var);

  auto optimizerInputs = optimizer->getOptimizerInputs(var);

  // If there is no weight gradient, we assume that the gradient has been
  // forced to zero somewhere else in the backwards pass
  bool updaterAvailable = getMainGraph().getTensors().contains(
      inputIds.at(VarUpdateWithUpdaterOp::getUpdaterInIndex()));

  if (updaterAvailable) {

    // create the required optimizer tensors as needed
    for (auto opt : optimizerInputs) {
      auto optId   = std::get<0>(opt);
      auto optInfo = std::get<1>(opt);
      ensureOptimizerTensorCreated(optId, optInfo);
    }

    OpId opId =
        getMainGraph().moveIntoGraph(optimizer->createOp(var, getMainGraph()));

    getMainGraph().connectInputs(InputVecWrapper(inputIds), opId);
    growVarUpdateOpInternal(opId);
  }
}

void Ir::ensureOptimizerTensorCreated(const TensorId &optId,
                                      const TensorInfo &info) {
  if (!getTensors().contains(optId)) {

    getTensors().addStream(optId, info);
    Tensor &optTensor = *getTensors().get(optId);
    optimizer->setTensorData(optTensor);

    // optimizer tensors are a special type of stream which is broadcast
    optTensor.setReplicatedStreamMode(Tensor::ReplicatedStreamMode::Broadcast);
  }
}

void Ir::growVarUpdateOpInternal(OpId opId) {
  Op *op           = getMainGraph().getOps()[opId].get();
  auto varUpdateOp = dynamic_cast<VarUpdateOp *>(op);
  if (varUpdateOp == nullptr) {
    throw internal_error("Op {} expected to be a VarUpdateOp", op->str());
  }
  TensorId updatedVarId = getUpdatedVarId(varUpdateOp->getVarId());
  std::vector<TensorId> outputs{updatedVarId};
  getMainGraph().connectOutputs(OutputVecWrapper(outputs), opId);
  op->setup();
  op->inheritPlacementAttributes(false);
}

std::set<Op *> Ir::getTrainTargetOps() const {
  std::set<Op *> trainTargets;
  for (auto &op : getMainGraph().getOps()) {
    if (op.second->isConvertibleTo<VarUpdateOp>()) {
      trainTargets.insert(op.second.get());
    }
  }
  return trainTargets;
}

void Ir::setFinalLoss(const TensorId &loss) {
  logging::ir::info("Growing final loss");

  if (getMainGraph().getTensors().contains(loss)) {
    if (getMainGraph().getTensors().get(loss)->info.nelms() > 1) {
      throw error("Loss tensor, '{}', must be a scalar tensor", loss);
    }

    // The final Loss Op is the only Op which (we say) has both
    // paths to and from
    auto finalLossOp      = getTensors().get(loss)->getProducer();
    finalLossOp->toLoss   = PathToLoss::Yes;
    finalLossOp->fromLoss = PathFromLoss::Yes;
    finalLossId           = loss;
    finalLossOpId         = finalLossOp->id;

    logging::ir::trace("Final loss Op id set to {}", finalLossOpId);
  } else {
    throw error("Could not find loss tensor '{}' in main graph tensors", loss);
  }

  constructedFinalLoss = true;
}

TensorId Ir::getFinalLossId() const { return finalLossId; }

void Ir::append(std::stringstream &ss) const {
  ss << "\n";

  int i = 0;
  for (auto graph : getGraphSchedule()) {
    if (i > 0) {
      ss << "============================================================\n";
    }
    i += 1;

    if (graph->id.str() != "") {
      ss << graph->id.str() << ":"
         << "\n";
    }

    for (auto &op : graph->getOpSchedule({})) {
      op->append(ss);
    }
  }
}

namespace {

void setGraphIrName(const std::string &name, std::stringstream &ss) {
  if (name.find("BuilderGraph_") != std::string::npos) {
    ss << "\"maingraph\" :[";
  } else {
    ss << "\"" << name << "\" :[";
  }
}

} // namespace

void Ir::serialise(SerialiseFormat format,
                   std::stringstream &ss,
                   bool useScheduler) const {

  auto getGraphs = [this, useScheduler]() {
    if (useScheduler) {
      return getGraphSchedule();
    } else {
      std::vector<const Graph *> result;
      for (auto &id_graph : graphs) {
        auto graph = id_graph.second.get();
        result.push_back(graph);
      }
      return result;
    }
  };

  auto getOps = [useScheduler](auto *graph) {
    if (useScheduler) {
      return graph->getOpSchedule({});
    } else {
      std::vector<Op *> result;
      for (auto &id_op : graph->getOps()) {
        auto op = id_op.second.get();
        result.push_back(op);
      }
      return result;
    }
  };

  // TODO use the format to seralize the ir
  (void)format;

  ss << "{";

  bool firstGraph = true;
  for (auto graph : getGraphs()) {

    if (!firstGraph)
      ss << ",";

    if (firstGraph)
      setGraphIrName(graph->getIr().getModel().graph().name(), ss);
    else
      ss << "\"" << graph->id.str() << "\" :[";

    bool firstOp = true;
    for (auto &op : getOps(graph)) {

      if (!firstOp)
        ss << ",";

      op->toJSON(ss);

      firstOp = false;
    }

    ss << "]";

    firstGraph = false;
  }

  ss << "}";
}

int Ir::getDefaultOpsetVersion(const std::string &domain) const {
  if (domain == Domain::ai_onnx) {
    return defaultAiOnnxOpset;
  } else if (domain == Domain::ai_onnx_ml) {
    return defaultAiOnnxMlOpset;
  } else if (domain == Domain::ai_graphcore) {
    return defaultAiGraphcoreOpset;
  } else {
    throw error("No default opset version defined for domain \'{}\'", domain);
  }
}

int Ir::getOpSetVersionFromModel(const std::string &node_domain) const {

  // If the node.domain is blank it means the default ai.onnx
  auto domain = node_domain;
  if (domain == "") {
    domain = Domain::ai_onnx;
  }

  // Get the version of the opset from the model based on the domain
  int version    = 0;
  auto opsetList = getModel().opset_import();
  for (auto &opset : opsetList) {

    std::string opset_domain;
    if (opset.has_domain() == false || opset.domain() == "") {
      opset_domain = Domain::ai_onnx;
    } else {
      opset_domain = opset.domain();
    }

    if (domain == opset_domain) {

      auto opset_version = static_cast<int>(opset.version());

      // If the same domain is mentioned multiple times find the largest
      if (opset_version > version)
        version = opset_version;
    }
  }

  // If the version has not be set use the default
  if (version == 0) {
    version = getDefaultOpsetVersion(domain);
  }

  return version;
}

unsigned Ir::getMaxVirtualGraphId() const {
  unsigned maxVirtualGraphId = 1;
  unsigned replGraphCount =
      static_cast<unsigned>(getSessionOptions().replicatedGraphCount);
  unsigned numIPUs = static_cast<unsigned>(deviceInfo->getNumIpus());
  if (getSessionOptions().enableReplicatedGraphs) {
    if (numIPUs % replGraphCount != 0) {
      throw error("For replicated graphs, the number of IPUs must be divisible "
                  "by the replication factor.");
    } else {
      maxVirtualGraphId = numIPUs / replGraphCount;
    }
  } else {
    maxVirtualGraphId = numIPUs;
  }
  return maxVirtualGraphId;
}

std::vector<GradNonGradPair> Ir::growLossGradients() {
  // Create a constant scalar tensor of value '1' with same type as loss.
  // This is the start of the backpropagated gradients.
  TensorInfo gradStarterInfo = getTensors().get(getFinalLossId())->info;
  switch (gradStarterInfo.dataType()) {
  case DataType::FLOAT: {
    std::vector<float> gradStarterData(1, 1.0);
    getTensors().addConstInit(getGradId(getFinalLossId()),
                              gradStarterInfo,
                              reinterpret_cast<void *>(gradStarterData.data()));
    break;
  }
  case DataType::FLOAT16: {
    std::vector<float> floatData(1, 1.0);
    std::vector<char> gradStarterData(2);
    poplar::copyFloatToDeviceHalf(
        poplar::Target(), floatData.data(), gradStarterData.data(), 1);
    getTensors().addConstInit(getGradId(getFinalLossId()),
                              gradStarterInfo,
                              reinterpret_cast<void *>(gradStarterData.data()));
    break;
  }
  case DataType::INT16: {
    std::vector<int16_t> gradStarterData(1, 1);
    getTensors().addConstInit(getGradId(getFinalLossId()),
                              gradStarterInfo,
                              reinterpret_cast<void *>(gradStarterData.data()));
    break;
  }
  case DataType::INT32: {
    std::vector<int32_t> gradStarterData(1, 1);
    getTensors().addConstInit(getGradId(getFinalLossId()),
                              gradStarterInfo,
                              reinterpret_cast<void *>(gradStarterData.data()));
    break;
  }
  case DataType::INT64: {
    std::vector<int64_t> gradStarterData(1, 1);
    getTensors().addConstInit(getGradId(getFinalLossId()),
                              gradStarterInfo,
                              reinterpret_cast<void *>(gradStarterData.data()));
    break;
  }
  case DataType::UINT32: {
    std::vector<uint32_t> gradStarterData(1, 1);
    getTensors().addConstInit(getGradId(getFinalLossId()),
                              gradStarterInfo,
                              reinterpret_cast<void *>(gradStarterData.data()));
    break;
  }
  case DataType::UINT64: {
    std::vector<uint64_t> gradStarterData(1, 1);
    getTensors().addConstInit(getGradId(getFinalLossId()),
                              gradStarterInfo,
                              reinterpret_cast<void *>(gradStarterData.data()));
    break;
  }
  default: {
    throw error("Unexpected loss data-type, '{}'",
                gradStarterInfo.getDataTypeInfo()->name());
  }
  }

  auto finalLossOpFound = getMainGraph().getOps().find(finalLossOpId);
  if (finalLossOpFound != getMainGraph().getOps().end()) {
    std::vector<GradNonGradPair> pairs;
    auto finalLossOp = getMainGraph().getOp(finalLossOpId);
    for (Op *gradOp : growGradOps(finalLossOp)) {
      pairs.push_back({gradOp, finalLossOp});
    }
    return pairs;
  } else {
    throw error("Call to growLossGradients, but finalLossOpId not found");
  }
}

OpId Ir::getFinalLossOpId() const { return finalLossOpId; }

std::vector<const Graph *> Ir::getGraphSchedule() const {
  std::vector<const Graph *> sorted;
  std::set<const Graph *> seen;

  std::function<void(const Graph *)> scheduleGraph;
  scheduleGraph = [&](const Graph *graph) {
    // only try schedule a graph once
    if (seen.find(graph) == seen.end()) {
      seen.insert(graph);
    } else {
      return;
    }

    // add graph to schedule
    sorted.push_back(graph);

    // schedule all called graphs
    for (auto g : graph->getCalledGraphs()) {
      scheduleGraph(g);
    }
  };

  scheduleGraph(&getMainGraph());

  if (sorted.size() != graphs.size()) {
    throw error("Unable to schedule all graphs. {} != {}",
                sorted.size(),
                graphs.size());
  }

  return sorted;
}

bool Ir::hasRandomOps() const {
  for (auto op : getAllOps()) {
    if (op->requiresRandomSeed()) {
      return true;
    }
  }
  return false;
}

bool Ir::requiresRandomSeed() const {
  return (getSessionOptions().enableStochasticRounding || hasRandomOps());
}

void Ir::initRandomSeed() {
  // 1. create seed tensor
  TensorId seedId = GetRandomSeedOp::getStreamedSeedTensorId();
  DataType dtype  = DataType::UINT32;
  TensorInfo info(dtype, {2});
  getTensors().addStream(seedId, {dtype, {2}});
  Tensor &seedTensor = *getTensors().get(seedId);
  seedTensor.setReplicatedStreamMode(Tensor::ReplicatedStreamMode::Replicate);

  // 2. Set initial value (from clock)
  uint64_t init = std::chrono::system_clock::now().time_since_epoch().count();
  setRandomSeedValue(init);

  // 3. create GetRandomSeed op and connect to seed tensor
  Op::Settings settings(getMainGraph(), "");
  auto getSeedOp_up = std::make_unique<GetRandomSeedOp>(
      Onnx::CustomOperators::GetRandomSeed, settings);
  auto getSeedOp = getSeedOp_up.get();

  auto allOtherOps                  = getOpSchedule({});
  bool allOtherOpsHavePipelineStage = true;
  for (auto op : allOtherOps) {
    if (!op->hasPipelineStage()) {
      allOtherOpsHavePipelineStage = false;
    }
  }
  getMainGraph().moveIntoGraph(std::move(getSeedOp_up));
  if (virtualGraphsEnabled()) {
    getSeedOp->setVirtualGraphId(0);
    if (getSessionOptions().enablePipelining && allOtherOpsHavePipelineStage) {
      getSeedOp->setPipelineStage(0);
    }
  }
  getSeedOp->connectInTensor(getSeedOp->getSeedInIndex(), seedId);
  TensorId updatedSeedId = GetRandomSeedOp::getUpdatedSeedTensorId();
  getSeedOp->createAndConnectOutTensor(
      GetRandomSeedOp::getUpdatedSeedOutIndex(), updatedSeedId);
  getSeedOp->setup();

  // 4. hook up to fwd Random ops
  for (auto op : getOpSchedule({})) {
    if (op->requiresRandomSeed()) {
      op->connectInTensor(op->getSeedInIndex(), updatedSeedId);
    }
  }
}

void Ir::setRandomSeedValue(uint64_t seedValue) {
  logging::ir::info("Setting the random seed to {}", seedValue);
  TensorId seedId    = GetRandomSeedOp::getStreamedSeedTensorId();
  Tensor *seedTensor = getTensor(seedId);
  std::vector<char> seedData(seedTensor->info.nbytes());
  *reinterpret_cast<uint64_t *>(seedData.data()) = seedValue;
  if (seedTensor->hasTensorData()) {
    seedTensor->tensorData()->resetData(seedTensor->info, seedData.data());
  } else {
    seedTensor->setTensorData(seedTensor->info, seedData.data());
  }
}

std::vector<Op *> Ir::getOpSchedule(const OpsBeforeKey &gCons) const {
  std::vector<Op *> sorted;
  std::set<const Graph *> addedGraphs;

  std::function<void(const Graph *)> addGraph;
  addGraph = [&](const Graph *graph) {
    // Only add each graph once
    if (addedGraphs.find(graph) != addedGraphs.end()) {
      return;
    }
    addedGraphs.insert(graph);

    // Add each op in the graph
    for (auto op : graph->getOpSchedule(gCons)) {
      // If the op calls another graph
      // the ops in that graph should be scheduled first
      for (auto calledGraph : op->getCalledGraphs()) {
        addGraph(calledGraph);
      }

      sorted.push_back(op);
    }
  };

  // Start adding ops from the main graph
  addGraph(&getMainGraph());

  return sorted;
}

// Are the Ops with all the dependencies a DAG?
bool Ir::isSchedulable(const OpsBeforeKey &gCons) const {
  for (auto &id_graph : graphs) {
    if (!id_graph.second->isSchedulable(gCons)) {
      return false;
    }
  }
  return true;
}

Ir::ExecutionMode Ir::getExecutionMode() const { return executionMode; }

bool Ir::canInfer() const {
  return getExecutionMode() == ExecutionMode::Inference || canTrain();
}

bool Ir::canTrain() const {
  return getExecutionMode() == ExecutionMode::Training;
}

bool Ir::hasConstructedBackwards() const { return constructedBackwards; }

bool Ir::containsInitialisers() {
  return !(onnxModel->graph().initializer().empty());
}

bool Ir::tensorExistsInInitialisers(TensorId tId) const {
  for (int init_index = 0; init_index < onnxModel->graph().initializer_size();
       ++init_index) {
    if (onnxModel->graph().initializer(init_index).name() == tId) {
      return true;
    }
  }
  return false;
}

void Ir::applyUpdateInplacePrioritiesForIpu() {
  UpdateInplacePrioritiesForIpu pattern;

  for (auto &id_graph : graphs) {
    auto graph = id_graph.second.get();
    for (auto &id_op : graph->getOps()) {
      Op *op = id_op.second.get();
      if (!op->isExcludedFromPattern(&pattern)) {
        pattern.apply(op);
      }
    }
  }
}

void Ir::applyInplacePattern(Graph &graph) {

  logging::ir::debug("Applying Inplace Pattern to Graph \"{}\"", graph.id);

  Inplace inplace;

  // <0> the id of the Op to inplace
  // <1> the type of the inplace Op
  // <2> the priority of this inplacing
  using Triplet = std::tuple<OpId, OperatorIdentifier, float>;

  std::vector<Triplet> priorities;
  for (auto &id_op : graph.getOps()) {
    Op *op = id_op.second.get();

    // first see if the user has overriden the default priorities
    std::set<OpType> prioritized;
    for (auto ip : op->settings.inplacePriorityVeto) {
      OpType inplaceId = std::get<0>(ip);
      priorities.push_back({
          op->id,
          {
              Domain::ai_graphcore, // the domain (same for all inplace ops)
              inplaceId,            // the name of the Operator (OpId)
              1                     // version
          },
          std::get<1>(ip) // the priority value
      });
      prioritized.insert(inplaceId);
    }

    // for all the inplacers not in the user list, take the default
    for (auto ip : op->inplacePriorityDefault()) {
      OperatorIdentifier identifier = std::get<0>(ip);
      if (prioritized.count(identifier.type) == 0) {
        priorities.push_back({op->id, identifier, std::get<1>(ip)});
      }
    }
  }

  auto tripletComparator = [](const Triplet &a, const Triplet &b) {
    if (std::get<2>(a) - std::get<2>(b) != 0.0f) {
      return std::get<2>(a) > std::get<2>(b);
    }
    // if same priority, fall back to ID to keep it deterministic
    return std::get<0>(a) > std::get<0>(b);
  };

  if (priorities.size() != 0) {

    // sort in decreasing order of priority,
    std::sort(priorities.begin(), priorities.end(), tripletComparator);

    // removing all negative priorities. We use std::lower_bound
    // instead of std::find_if, taking advantage of the fact that priorities
    // are sorted at this point.

    // (1) we create a "pivot" with priority 0
    Triplet zeroPriority      = priorities[0];
    std::get<2>(zeroPriority) = 0.;

    // (2) we find the first elememts in priorities which is not less than the
    // pivot, and erase all elements from there to the end. Note that
    // priority 0 elements will be removed.
    auto found = std::lower_bound(
        priorities.begin(), priorities.end(), zeroPriority, tripletComparator);
    priorities.erase(found, priorities.end());

    // we keep track of which ops have already been inplaced
    std::set<OpId> inplacedAlready;

    for (auto &ip : priorities) {
      OpId id                       = std::get<0>(ip);
      OperatorIdentifier identifier = std::get<1>(ip);

      // check that the op has not already been inplaced
      auto inplaced_already_it = inplacedAlready.find(id);
      if (inplaced_already_it != inplacedAlready.end()) {
        std::ostringstream oss;
        oss << "[Inplacing] The Op being considered for inplacing, " << id
            << ", is already inplace.";
        logging::pattern::debug(oss.str());
        continue;
      }

      Op *op = graph.getOps().at(id).get();

      if (op->isExcludedFromPattern(&inplace)) {
        std::ostringstream oss;
        oss << "[Inplacing] The Op being considered for inplacing, "
            << op->str() << ", is excluded from the Inplacing Pattern.";
        logging::pattern::debug(oss.str());
        continue;
      }

      // This is where the hard work of inplacing gets done: if op is inplaced,
      // what additional topological constraints are needed to ensure
      // correctness?
      OpsBeforeKey newTopoCons = inplace.getNewTopoCons(op, identifier);

      // beforeProducesOutput flag is used to prevent inplacing if any of the
      // new constraints requried to inplace a node has a before node that
      // produces an output of the graph. this is prevented because if the graph
      // is executed using a call op, then the out from the nodes are copied
      // after all the nodes of the sub graph have executed. this would cause
      // the inplaced data to be corrupted even if the constraints are in place
      // as the tensor output copy is delayed.
      bool beforeProducesOutput = false;
      for (auto &before_after : newTopoCons) {
        auto &befores = before_after.second;

        for (auto &before : befores) {
          if (before->producesGraphOutput()) {
            beforeProducesOutput =
                true; // before node of the topocon constraint produces output
            std::ostringstream oss;
            oss << "[Inplacing] " << op->str()
                << ", Excluded due to the required topological constraint with "
                   "output node, "
                << before->str();
            logging::pattern::debug(oss.str());
            break;
          }
        }
        if (beforeProducesOutput)
          break;
      }
      if (beforeProducesOutput) {
        continue;
      }

      ExternOpTensorBundle eot_bun(op, op->getInplaceVariant(identifier));
      const Op *inplaceOp = eot_bun.getOp();

      // If any of the new topological constraints start at an Op whose inputs
      // cannot be modified (consume anchors, certain recomputatation Ops, etc),
      // then we cannot inplace op. This is because a constraint op -> op2 is
      // inserted if and only if op2 modifies an input of op, which is exactly
      // the case we must avoid if op->inputsUnmodifiable().
      std::vector<const Op *> blockingTopoConFroms;
      if (inplaceOp->modifies()) {
        if (op->inputsUnmodifiable()) {
          blockingTopoConFroms.push_back(op);
        }
        auto opFound = newTopoCons.find(op);
        if (opFound != newTopoCons.end()) {
          for (const auto before : opFound->second) {
            if (before->inputsUnmodifiable()) {
              blockingTopoConFroms.push_back(before);
            }
          }
        }
      }
      for (const auto &after_befores : newTopoCons) {
        const auto after = after_befores.first;
        for (const auto before : after_befores.second) {
          if (after->modifies() && before->inputsUnmodifiable()) {
            blockingTopoConFroms.push_back(before);
          }
        }
      }
      if (!blockingTopoConFroms.empty()) {
        std::ostringstream oss;
        oss << "[Inplacing] The Op being considered for inplacing, "
            << op->str() << " cannot be inplaced because it would effect "
            << "inputs which cannot be modified. "
            << "In particular, "
            << "the following Ops have unmodifiable inputs "
            << "which prevent inplacing:";
        for (const auto before : blockingTopoConFroms) {
          oss << "\n     " << before->id << " : "
              << before->getInputsUnmodifiableString();
        }
        logging::pattern::debug(oss.str());
        continue;
      }

      // finally, we check if there are cycles with the new topological
      // constraints
      if (!graph.isSchedulable(newTopoCons)) {
        std::ostringstream oss;
        oss << "[Inplacing] The new topological constraints prevent Op "
            << op->id << " from being inplaced, as they would created a cycle ";
        logging::pattern::debug(oss.str());
        continue;
      }

      {
        std::ostringstream oss;
        oss << "[Inplacing] Inplacing Op " << op->str();
        logging::pattern::debug(oss.str());
        inplacedAlready.insert(op->id);
        inplace.apply(op, identifier, newTopoCons);
      }
    }
  }
  logging::pattern::trace("Completed Inplacing Pattern");
}

Op &Ir::getSubgraphAnchorPlaceholder() {
  static std::unique_ptr<Op> subgraphAnchorPlaceholder = std::unique_ptr<Op>(
      new PlaceholderOp({"TempAnchorDomain", "TempAnchorType", 1},
                        Op::Settings{getMainGraph(), "TempAnchorName"}));

  return *subgraphAnchorPlaceholder.get();
}

std::vector<TensorId> Ir::getTensorIds(TensorType tensor_type) const {
  std::vector<TensorId> result;

  for (auto &id_graph : graphs) {
    auto graph = id_graph.second.get();
    auto ids   = graph->getTensors().getIds(tensor_type);
    result.reserve(result.size() + ids.size());
    result.insert(result.end(), ids.begin(), ids.end());
  }

  return result;
}

Tensor *Ir::getTensor(const TensorId &tensor_id) const {
  for (auto &id_graph : graphs) {
    auto graph = id_graph.second.get();
    if (graph->getTensors().contains(tensor_id)) {
      return graph->getTensors().get(tensor_id);
    }
  }

  throw error("no Ir::Tensor with TensorId " + tensor_id +
              ", in Ir::getTensor(..) ");
}

bool Ir::containsTensor(const TensorId &tensor_id) const {
  for (auto &id_graph : graphs) {
    auto graph = id_graph.second.get();
    if (graph->getTensors().contains(tensor_id)) {
      return true;
    }
  }
  return false;
}

std::vector<TensorId> Ir::getGraphInputIds() const {
  std::vector<TensorId> result;

  for (auto &id_graph : graphs) {
    auto graph = id_graph.second.get();
    auto &ids  = graph->getInputIds();
    result.reserve(result.size() + ids.size());
    result.insert(result.end(), ids.begin(), ids.end());
  }

  return result;
}

const Tensors &Ir::getTensors() const { return getMainGraph().getTensors(); }
Tensors &Ir::getTensors() { return getMainGraph().getTensors(); }

const Graph &Ir::getMainGraph() const { return getGraph(GraphId::root()); }
Graph &Ir::getMainGraph() { return getGraph(GraphId::root()); }

Graph &Ir::getGraph(const GraphId &graphId) const {
  return *graphs.at(graphId);
}

std::vector<const Graph *> Ir::getAllGraphs() const {
  std::vector<const Graph *> allGraphs;
  for (auto &id_graph : graphs) {
    allGraphs.push_back(id_graph.second.get());
  }
  return allGraphs;
}

bool Ir::hasGraph(const GraphId &graphId) const {
  return graphs.find(graphId) != graphs.end();
}

Graph &Ir::createGraph(const GraphId &graphId) {
  logging::ir::trace("Creating Graph with id \"{}\"", graphId);
  auto found = graphs.find(graphId);
  if (found != graphs.end()) {
    throw error("Graph({}) is already in Ir", graphId);
  }

  graphs.insert({graphId, std::make_unique<Graph>(*this, graphId)});
  return getGraph(graphId);
}

std::map<OpId, std::unique_ptr<Op>> &Ir::getMainGraphOps() {
  return getMainGraph().getOps();
}

const std::map<OpId, std::unique_ptr<Op>> &Ir::getMainGraphOps() const {
  return getMainGraph().getOps();
}

std::vector<Op *> Ir::getAllOps() const {
  std::vector<Op *> ops;
  for (auto &graph : graphs) {
    for (auto &op : graph.second->getOps()) {
      ops.push_back(op.second.get());
    }
  }
  return ops;
}

Tensors &Ir::getMainGraphTensors() { return getMainGraph().getTensors(); }

const Tensors &Ir::getMainGraphTensors() const {
  return getMainGraph().getTensors();
}

uint32_t Ir::getAndIncrementDropoutSeedModifier() {
  dropoutSeedModifier += 1;
  return dropoutSeedModifier;
}

void Ir::setRemoteBufferInfo(RemoteBufferId id, RemoteBufferInfo info) {
  remoteBufferInfoMap.insert({id, info});
}

const RemoteBufferInfo Ir::getRemoteBufferInfo(RemoteBufferId id) const {
  return remoteBufferInfoMap.at(id);
}

const std::map<RemoteBufferId, RemoteBufferInfo>
Ir::getAllRemoteBufferInfos() const {
  return remoteBufferInfoMap;
}

TensorId Ir::createIntermediateTensorId(const TensorId &base_id) {
  auto temp_id =
      logging::format("{}__t{}", base_id, intermediate_tensor_counter);
  logging::ir::trace("Generating tensor id {}", temp_id);
  ++intermediate_tensor_counter;
  return temp_id;
}

TensorId
Ir::createBatchSliceTensorId(TensorId base_id, unsigned s, unsigned e) {
  auto slice_id = logging::format(
      "{}__bs{}_{}_{}", base_id, s, e, intermediate_tensor_counter);
  logging::ir::trace("Generating tensor id {}", slice_id);
  ++intermediate_tensor_counter;
  return slice_id;
}

TensorId Ir::createBatchConcatTensorId(TensorId base_id) {
  auto concat_id =
      logging::format("{}__cc{}", base_id, intermediate_tensor_counter);
  logging::ir::trace("Generating tensor id {}", concat_id);
  ++intermediate_tensor_counter;
  return concat_id;
}

GraphId Ir::createUniqueSubgraphId(GraphId base_id) {
  auto next_id =
      logging::format("{}_subgraph({})", base_id, subgraph_id_counter);
  ++subgraph_id_counter;
  return next_id;
}

} // namespace popart
