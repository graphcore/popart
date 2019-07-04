#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <poponnx/builder.hpp>
#include <poponnx/ces/constexpr.hpp>
#include <poponnx/ces/onnxconstexpr.hpp>
#include <poponnx/chains.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/intervals.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/op/loss.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/scheduler.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/topocons.hpp>
#include <poponnx/util.hpp>

// The transformations
#include <poponnx/recompute.hpp>
#include <poponnx/transforms/auto_virtual_graph.hpp>
#include <poponnx/transforms/interipucopy.hpp>
#include <poponnx/transforms/mergecopies.hpp>
#include <poponnx/transforms/mergevarupdates.hpp>
#include <poponnx/transforms/prune.hpp>
#include <poponnx/transforms/subgraphoutline.hpp>

// The layers required to construct the backwards pass
#include <poponnx/op/batchnorm.hpp>
#include <poponnx/op/ipucopy.hpp>
#include <poponnx/op/placeholder.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/op/varupdate.hpp>

#include <poponnx/patterns/inplace.hpp>
#include <poponnx/patterns/updateinplaceprioritiesforipu.hpp>

#include <poponnx/dotvisualizer.hpp>

namespace poponnx {

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

const onnx::ModelProto &Ir::getModel() const { return *onnxModel; }

std::vector<Tensor *> Ir::optimizerTensors() const {
  std::vector<Tensor *> optTensors;
  if (optimizer.get() != nullptr) {
    for (auto &id_info : optimizer->tensorInfos()) {
      // some tensors might have been removed,
      // check they exist before calling getTensors().get(...)
      if (getTensors().contains(id_info.first)) {
        optTensors.push_back(getTensors().get(id_info.first));
      }
    }
  }
  return optTensors;
}

// the rule followed : a Stream tensor which is not an
// optimizer tensor is a streamed data tensor
std::vector<Tensor *> Ir::dataStreamTensors() const {
  std::vector<Tensor *> dsTensors;
  std::map<TensorId, TensorInfo> optTensorInfo;
  if (optimizer != nullptr) {
    optTensorInfo = optimizer->tensorInfos();
  }
  for (TensorId id : getTensors().getIds(TensorType::Stream)) {
    if (optTensorInfo.find(id) == optTensorInfo.end()) {
      dsTensors.push_back(getTensors().get(id));
    }
  }
  return dsTensors;
}

void Ir::updateOptimizer(const Optimizer *newOptimizer) {
  if (optimizer.get() == nullptr) {
    throw error("ILE: cannot update optimizer before it is set");
  }
  if (!optimizer->validReplacement(newOptimizer)) {
    throw error("This Optimizer of type " + newOptimizer->type_s() +
                " is not a valid replacement for optimizer of type " +
                optimizer->type_s());
  }
  optimizer = newOptimizer->clone();
  optimizer->resetTensorDatas(getMainGraph());
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

IrBundle::IrBundle(const onnx::ModelProto &modelProto_,
                   const InputShapeInfo &inputShapeInfo_,
                   const DataFlow &dataFlow_,
                   const std::vector<Loss *> &losses_,
                   const Optimizer *optimizer_,
                   DeviceInfo &deviceInfo_,
                   const SessionOptions &userOptions_,
                   const Patterns &patterns_)
    : modelProto(modelProto_), inputShapeInfo(inputShapeInfo_),
      dataFlow(dataFlow_), losses(losses_), optimizer(optimizer_),
      deviceInfo(deviceInfo_), userOptions(userOptions_), patterns(patterns_) {}

Ir::Ir() : onnxModel(nullptr) {
  graphs.insert(
      {GraphId::root(), std::make_unique<Graph>(*this, GraphId::root())});
}

void Ir::setOnnxModel(const onnx::ModelProto &model) {
  onnxModel.reset(new onnx::ModelProto(model));
}

void Ir::setDataFlow(const DataFlow &df) {
  // Inference and evaluation modes require an anchor
  if (!canTrain() && df.nAnchors() == 0) {
    throw error("User must specify an anchor tensor when doing inference or "
                "evalulation.");
  } else {
    dataFlow = df;
  }
}

void Ir::setUserOptions(const SessionOptions &flags) { userOptions = flags; }
void Ir::setInputShapeInfo(const InputShapeInfo &info) {
  inputShapeInfo = info;
}

void Ir::setPatterns(const Patterns &p) { patterns = p; }

void Ir::removeIsolatedTensors() { getTensors().removeIsolated(); }

void Ir::setExecutionMode(const ExecutionMode &mode) { executionMode = mode; }

void Ir::setLosses(const std::vector<Loss *> &_losses) {
  losses.clear();
  for (auto &l : _losses) {
    losses.emplace_back(l->clone());
  }
}

void Ir::setOptimizer(const Optimizer *o) {
  if (o) {
    optimizer = o->clone();

    for (auto &id_info : optimizer->tensorInfos()) {
      TensorId id     = id_info.first;
      TensorInfo info = id_info.second;
      getTensors().addStream(id, info);

      Tensor *tensor = getTensors().get(id);
      optimizer->setTensorData(tensor);

      // optimizer tensors are a speical type of stream which is broadcast
      tensor->setReplicatedStreamMode(Tensor::ReplicatedStreamMode::Broadcast);
    }
  }
}

void Ir::setDeviceInfo(DeviceInfo &di) { deviceInfo = &di; }

const DeviceInfo *Ir::getDeviceInfo() { return deviceInfo; }

void Ir::logIr() {
  std::stringstream ss2;
  append(ss2);
  logging::ir::info(ss2.str());
}

void Ir::verifyOpOutputConnectivity(const Graph &graph) const {
  logging::ir::info("Checking op output tensor producers");

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
  logging::ir::info("Checking op input tensor consumers");

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
  logging::ir::info("Checking tensor producer outputs");

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
      throw error("Tensor {} is a variable tensor, but has op {} as a producer",
                  tensor->str(),
                  op->str());
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
  logging::ir::info("Checking tensor consumer inputs");

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

bool Ir::isCandidateForConstExprFolding(const Tensor &tensor) const {
  auto tt = tensor.tensorType();

  if (canTrain()) {
    if (tt != TensorType::Const) {
      return false;
    }
  } else {
    // evalulation or inference
    if (tt != TensorType::Const && tt != TensorType::Variable) {
      return false;
    }
  }
  return true;
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
//    inference and eval modes
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
  setDeviceInfo(gb.deviceInfo);

  if (isPrepared) {
    throw error("Ir::prepare called more than once");
  }

  // Require gb.losses.empty() => !gb.optimizer
  if (gb.losses.empty() && gb.optimizer) {
    throw error("An optimizer is set without any losses");
  }

  if (gb.optimizer) {
    setExecutionMode(ExecutionMode::TRAINING);
  } else if (gb.losses.empty()) {
    setExecutionMode(ExecutionMode::INFERENCE);
  } else {
    setExecutionMode(ExecutionMode::EVALUATION);
  }

  setDataFlow(gb.dataFlow);
  setUserOptions(gb.userOptions);
  setInputShapeInfo(gb.inputShapeInfo);
  setPatterns(gb.patterns);
  setOnnxModel(gb.modelProto);

  setLosses(gb.losses);

  confirmNoReservedIds();

  registerInputTensors();

  logging::ir::info("Patterns : {}", patterns);
  // todo : validate the selected patterns

  // construct the forward pass from ONNX,
  constructForwards();

  // Check virtual graph settings and annotations are consistent
  verifyVirtualGraphIds(false);

  dotCheckpoint(DotCheck::FWD0);

  for (auto &id_graph : graphs) {
    auto &graph = getGraph(id_graph.first);
    applyPreAliasPatterns(graph);
  }
  dotCheckpoint(DotCheck::FWD1);

  enableTransform(AutoVirtualGraph::id(),
                  userOptions.autoVirtualGraph &&
                      userOptions.enableVirtualGraphs);
  applyTransform(AutoVirtualGraph::id(), getMainGraph());

  if (canEvaluate()) {
    growFinalLoss();
    updateVertices();
    setNEdgesToLoss();
  }

  if (autoRecomputationEnabled() && getMainGraph().hasUserRecomputeOps()) {
    throw error("A mixture of auto and manual recomputaion is not supported");
  }

  // tensors with no producer and no consumers are removed
  // at this point. We may want something more subtle.
  removeIsolatedTensors();

  setOptimizer(gb.optimizer);

  updateVertices();
  if (canTrain()) {
    constructBackwards();
  }

  updateVertices();
  dotCheckpoint(DotCheck::BWD0);

  // confirm that all the anchor names provided
  // are indeed real tensor names. This is a check
  // that the user has not provided incorrect names.
  // We allow duplicates.
  validateAnchors();
  applyTransform(Prune::id(), getMainGraph());

  for (auto &id_graph : graphs) {
    auto &graph = getGraph(id_graph.first);
    applyPreAliasPatterns(graph);
  }

  if (canEvaluate()) {
    setNEdgesToLoss();
  }

  // tensors with no producer and no
  // consumers are removed at this point.
  removeIsolatedTensors();
  updateVertices();

  auto updateTrainTargetOps = [this]() {
    // reset the trainTargetOps, updated by MergeVarUpdates
    trainTargetOps.clear();
    for (auto &op : getMainGraph().getOps()) {
      if (op.second->isConvertibleTo<VarUpdateOp>()) {
        trainTargetOps.insert(op.second.get());
      }
    }
    updateVertices();
  };

  switch (userOptions.mergeVarUpdate) {

  case (MergeVarUpdateType::All): {
    enableTransform(MergeAllVarUpdates::id(), true);
    applyTransform(MergeAllVarUpdates::id(), getMainGraph());
    updateTrainTargetOps();
    break;
  }
  case (MergeVarUpdateType::AutoTight): {
    enableTransform(MergeTightThreshold::id(), true);
    applyTransform(MergeTightThreshold::id(), getMainGraph());
    updateTrainTargetOps();
    break;
  }
  case (MergeVarUpdateType::AutoLoose): {
    enableTransform(MergeLooseThreshold::id(), true);
    applyTransform(MergeLooseThreshold::id(), getMainGraph());
    updateTrainTargetOps();
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

  // Add internal ops to copy tensors between ipu's as needed
  applyTransform(InterIpuCopy::id(), getMainGraph());
  applyTransform(MergeCopies::id(), getMainGraph());
  updateVertices();

  dotCheckpoint(DotCheck::PREALIAS);

  // outlining makes Phase of Vertices meaningless as matches
  // can contain Ops from different Pphase. We should not
  // run updateVertices after this pass
  if (getSessionOptions().enableOutlining) {
    applyTransform(SubgraphOutline::id(), getMainGraph());
  }

  if (autoRecomputationEnabled()) {
    updateVertices();
    logging::transform::info("Auto-annotating Ops for recomputation");
    recompute::autoAnnotate(getMainGraph(),
                            getSessionOptions().autoRecomputation);
  }

  // Now, we apply the Patterns which can handle and create
  // topological constraints. Currently, this is only one
  // in-placing Pattern.
  if (patterns.isInPlaceEnabled()) {
    // Update the inplace priorities of ops before inplacing
    if (patterns.isUpdateInplacePrioritiesForIpuEnabled()) {
      applyUpdateInplacePrioritiesForIpu();
    }
    for (auto &id_graph : graphs) {
      applyInplacePattern(*id_graph.second);
    }
  }

  dotCheckpoint(DotCheck::FINAL);

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

  updateVertices();

  verifyConstExprFolding();
  verifyConnectivity();
  verifyTensorIds();
  verifyVirtualGraphIds(true);
  verifyVertexAttributesOnlyInMain();
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
  logging::ir::debug("Verifying virtual graph id consistency");
  // All non-IpuCopyOps, sorted by virtual graph id (-1 if not set)
  std::map<int64_t, std::vector<Op *>> vgraphs;
  for (auto &id_op : getMainGraph().getOps()) {
    if (!dynamic_cast<IpuCopyOp *>(id_op.second.get())) {
      int64_t vgid;
      if (id_op.second->getVirtualGraphId()) {
        vgid = *id_op.second->getVirtualGraphId();
      } else {
        vgid = -1;
      }

      auto found = vgraphs.find(vgid);
      if (found == vgraphs.end()) {
        vgraphs.insert({vgid, std::vector<Op *>{id_op.second.get()}});
      } else {
        found->second.push_back(id_op.second.get());
      }
    }
  }

  // a mix of annotated and not annotated Ops : suggests a problem
  if (vgraphs.count(-1) != 0 && vgraphs.size() > 1) {
    std::ostringstream errm;
    errm << "Either all Ops in the main graph must have their virtual "
         << "graph ids set, or none must. Histogram of Ops by virtual graph "
            "id\n";
    for (auto id_v : vgraphs) {
      errm << "  " << id_v.first << " : " << id_v.second.size() << "\n";
    }
    errm << "Ops with no virtual graph id :  \n";
    for (auto op : vgraphs[-1]) {
      errm << "  " << op->str() << "\n";
    }
    throw error(errm.str());
  }

  if (getSessionOptions().enableVirtualGraphs) {
    // only -1s, no Op has a virtual graph annotation : problem.
    if (vgraphs.size() == 1 && vgraphs.count(-1) != 0) {

      std::ostringstream errm;

      // no auto virtual graphing, the user should have annotated ops
      if (!getSessionOptions().autoVirtualGraph) {
        errm
            << "SessionOptions flag enableVirtualGraphs is true, "
            << "and flag autoVirtualGraph is false, "
            << "but no Ops have been annotated with virtual graph information. "
            << "This is an inconsistent combination. ";

        throw error(errm.str());
      }

      // auto virtual graphing, why has the auto-sharder not run?
      else if (postAutoVirtualGraphTransform) {
        errm
            << "SessionOptions flag enableVirtualGraphs is true, "
            << "and flag autoVirtualGraph is true, "
            << "but no Ops have been annotated with virtual graph information. "
            << "Moreover, the paramater postAutoVirtualGraphTransoform "
            << "is true, "
            << "so AutoVirtualGraph should have been run. "
            << "This is an inconsistent combination, possibly an internal "
            << "logic error has";

        throw error(errm.str());
      }
    }
  }

  else {
    for (auto vgid_ops : vgraphs) {
      auto vgid = vgid_ops.first;
      // enableVirtualGraphs is false, yet there is at least one Op with virtual
      // graph id set : suggests a problem
      if (vgid != -1) {
        auto op = vgid_ops.second.at(0);
        throw error("SessionOptions flag enableVirtualGraphs is false, but "
                    "{} has virtual graph id {}. This is inconsistent, "
                    "consider setting enableVirtualGraphs to true or removing "
                    "all virtual graph annotation from Ops. ",
                    op->str(),
                    op->getVirtualGraphId());
      }
    }
  }
}

void Ir::resetWeights(const onnx::ModelProto &modelProto) {
  auto &onnxGraph = modelProto.graph();

  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    if (!getTensors().contains(tenId)) {
      throw error("resetWeights, no tensor '" + tenId + "' in tensors");
    }
    auto tensor = getTensors().get(tenId);
    if (tensor->info != TensorInfo(initializer)) {
      throw error(
          "trying to reset weights using tensor with non matching tensor info");
    }
    tensor->tensorData()->resetData(initializer);
  }
}

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
    auto consumerId = fmt::format("{}@{}", node.op_type(), i);
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

    // need to look at the branch inputs for If node
    if (node.op_type() == Onnx::AiOnnx::OpSet9::If.type) {
      Attributes attr{node.attribute()};
      auto addBranchInputs = [&](std::string branchName) {
        auto branch = attr.getAttribute<Attributes::Graph>(branchName);
        for (int i = 0; i < branch.input_size(); i++) {
          auto inputId = branch.input(i).name();
          addConsumerType(inputId, node, i);
        }
      };
      addBranchInputs("then_branch");
      addBranchInputs("else_branch");
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

  std::set<TensorId> onnxInitializers;

  std::set<TensorId> unusedInitializers;

  for (const auto &initializer : onnxGraph.initializer()) {
    TensorId tenId = initializer.name();
    if (consumerTypes.find(tenId) == consumerTypes.end()) {
      logging::info("Not creating Tensor for unused initializer, {}", tenId);
      unusedInitializers.emplace(tenId);
    } else {
      // If inference or evaluation mode add initializers as constants if option
      // enabled
      if ((getExecutionMode() == ExecutionMode::INFERENCE ||
           getExecutionMode() == ExecutionMode::EVALUATION) &&
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
        throw error(
            "Request to create poponnx Stream Tensor {} failed, "
            "as it has no consumers in the ONNX GraphProto. "
            "If Tensor {} is only used as an input "
            "to a Loss, then it should not be included in the ONNX Model, "
            "but its TensorInfo should be in the InputShapeInfo object passed "
            "to the Ir/Session constructor.",
            id);
      }
      logCreationInfo("Stream", id);
      if (valueInfo.has_type() && valueInfo.type().tensor_type().has_shape()) {
        getTensors().addStream(id, TensorInfo(valueInfo.type()));
      } else {
        getTensors().addStream(id, inputShapeInfo.get(id));
      }
    }
  }

  // other true inputs are for the loss calculation (class labels, etc)
  for (const auto &loss : losses) {
    for (const auto &tenId : loss->getStreamTensorNames()) {
      // another loss might have already registered this tensor
      if (!getTensors().contains(tenId)) {
        getTensors().addStream(tenId, inputShapeInfo.get(tenId));
      } else {
        Tensor *tensorAlreadyPresent = getTensors().get(tenId);
        if (tensorAlreadyPresent->tensorType() != TensorType::Stream) {
          throw error("type mismatch for tensor " + tenId);
        }
      }
    }
  }
}

void Ir::validateAnchors() const {
  for (TensorId id : dataFlow.anchors()) {
    if (!getTensors().contains(id)) {
      std::stringstream ss;
      ss << "Anchor tensor `" << id << "' not in tensors. ";
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
      if (pattern->matches(op)) {
        if (!pattern->touchesAnchored(op)) {
          logging::pattern::debug("Applying pattern {} to {}",
                                  pattern->getPatternName(),
                                  op->debugName());
          result |= pattern->apply(op);
        }
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

bool Ir::isAnchored(const TensorId &tenId) const {
  return dataFlow.isAnchored(tenId);
}

void Ir::constructForwards() {
  constructFromOnnxGraph(onnxModel->graph(), {});
  for (auto &id_op : getMainGraph().getOps()) {
    auto op      = id_op.second.get();
    op->fromLoss = PathFromLoss::No;
  }
}

Graph &Ir::constructFromOnnxGraph(const onnx::GraphProto &graph,
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

Op *Ir::growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum) {

  std::unique_ptr<poponnx::Op> gradSum =
      OpManager::createOp(Domain::ai_onnx,
                          "Sum",
                          getOpSetVersionFromModel(Domain::ai_onnx),
                          getMainGraph(),
                          "GradSum");

  if (getSessionOptions().enableVirtualGraphs) {

    // Count which vgraph's the producer ops are on.
    std::map<int64_t, int64_t> vgraphIdMap;
    for (auto &t : toSum) {
      boost::optional<int64_t> vgraphId = t->getProducer()->getVirtualGraphId();
      if (vgraphId) {
        vgraphIdMap[*vgraphId]++;
      }
    }

    // Find the vgraph id with the most occurrences.
    auto it = std::max_element(vgraphIdMap.begin(),
                               vgraphIdMap.end(),
                               [](const std::pair<int64_t, int64_t> &p1,
                                  const std::pair<int64_t, int64_t> &p2) {
                                 return p1.second < p2.second;
                               });

    gradSum->setVirtualGraphId(it->first);
  }

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
  return op;
}

std::vector<Op *> Ir::growGradOps(Op *nonGradOp) {

  OpId nonGradOpId = nonGradOp->id;
  auto backOps     = nonGradOp->getGradOps();
  if (backOps.size() < 1) {
    throw error("Cannot get gradients for {}", nonGradOp->debugName());
  }
  std::vector<Op *> gradOps;
  for (auto &upop : backOps) {
    Op *gradOp    = upop.get();
    OpId gradOpId = getMainGraph().moveIntoGraph(std::move(upop));

    if (nonGradOp->settings.recomputeType == RecomputeType::RECOMPUTE &&
        autoRecomputationEnabled()) {
      throw error("Grad Ops should be grown before recompute annotation");
    }

    // connect inputs of gradOp
    {
      // inputs to gradOp (to populate in this scope):
      std::map<int, std::string> m_inputs;
      for (auto &inOutMapper : gradOp->gradInputInfo()) {

        int indexGrad     = inOutMapper.iGrad;
        int indexFwd      = inOutMapper.iNonGrad;
        GradOpInType type = inOutMapper.type;

        // the input at index 'indexGrad' to gradOp is
        switch (type) {
        //  (1) the INPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::IN: {
          if (!nonGradOp->input->hasIndex(indexFwd)) {
            throw error("Invalid configuration of gradOp {}. nonGradOp ({}) "
                        "INPUT {} is not defined ",
                        gradOp->debugName(),
                        nonGradOp->debugName(),
                        indexFwd);
          }
          m_inputs[indexGrad] = nonGradOp->input->tensor(indexFwd)->id;
          break;
        }

        //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
        case GradOpInType::OUT: {
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
        case GradOpInType::GRADOUT: {
          if (!nonGradOp->output->hasIndex(indexFwd)) {
            std::stringstream ss;
            ss << "No gradient for non-grad-op " << nonGradOp->debugName()
               << " at index " << indexFwd << '.'
               << " Could it be that the path along that index "
               << "did not lead to final loss, "
               << "in which case the gradient is zero?";
            throw error(ss.str());
          }
          m_inputs[indexGrad] =
              getGradId(nonGradOp->output->tensor(indexFwd)->id);
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

        if (!nonGradOp->input->tensor(nonGradIn)) {
          throw error("Invalid configuration of gradOp {}. nonGradOp ({}) "
                      "OUTPUT {} is not defined ",
                      gradOp->debugName(),
                      nonGradOp->debugName(),
                      nonGradIn);
        }

        TensorId inId  = nonGradOp->input->tensor(nonGradIn)->id;
        TensorId outId = getEdgeGradId(inId, nonGradOpId, nonGradIn);
        if (v_outputs.size() < gradOut + 1) {
          v_outputs.resize(gradOut + 1, "");
        }
        v_outputs[gradOut] = outId;
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
  auto found = partial.find(nonGrad->id);
  if (found == partial.end()) {
    partial.insert({nonGrad->id, {grad}});
  } else {
    partial[nonGrad->id].push_back(grad);
  }
  if (partial[nonGrad->id].size() == nonGrad->nEdgesToLoss) {
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
    throw error("ILE : index already present in OpGradRegistry::insert");
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
      "Updating all Vertices (toLoss, fromLoss, scheduledPreLoss)");

  // 1) get All Ops which have toLoss true, and backwards propagate
  std::vector<Op *> toLossFrontier;
  // 1) get All Ops which have fromLoss Yes, and backwards propagate
  std::vector<Op *> fromLossFrontier;
  for (auto &id_op : getMainGraph().getOps()) {
    Op *op = id_op.second.get();
    if (op->toLoss == PathToLoss::Yes) {
      toLossFrontier.push_back(op);
    }
    if (op->fromLoss == PathFromLoss::Yes) {
      fromLossFrontier.push_back(op);
    }
  }

  auto toLossVertices = backwardPropogate(toLossFrontier);
  for (Vertex *v : toLossVertices) {
    if (v->toLoss == PathToLoss::No) {
      throw error("ILE: Vertex {} deduced to have PathToLoss::Yes, but it "
                  "currently has PathToLoss::No",
                  v->str());
    }
    v->toLoss = PathToLoss::Yes;
  }

  auto fromLossVertices = forwardPropogate(fromLossFrontier);
  for (Vertex *v : fromLossVertices) {
    if (v->fromLoss == PathFromLoss::No) {
      throw error("ILE: Vertex {} deduced to have PathFromLoss::Yes, but has "
                  "PathFromLoss::No",
                  v->str());
    }
    v->fromLoss = PathFromLoss::Yes;
  }

  // set all Undefined to No
  for (auto &id_op : getMainGraph().getOps()) {
    auto setUnPaths = [](Vertex *v) {
      if (v->toLoss != PathToLoss::Yes) {
        v->toLoss = PathToLoss::No;
      }
      if (v->fromLoss != PathFromLoss::Yes) {
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
  // The first Op which is PathFromLoss::Yes, and all subsequent Ops, are
  // ScheduledPreLoss::No
  bool inFwd = true;
  for (auto op : getMainGraph().getOpSchedule({})) {
    if (op->fromLoss == PathFromLoss::Yes) {
      inFwd = false;
    }
    op->scheduledPreLoss = inFwd ? ScheduledPreLoss::Yes : ScheduledPreLoss::No;
  }

  // 3.2) scheduledPreLoss for Tensors
  for (auto op : getMainGraph().getOpSchedule({})) {
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

void Ir::setNEdgesToLoss() {

  if (isTesting()) {
    throw error("ILE: Call to setNEdgesToLoss() in Testing  mode is not valid");
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
  // initialised as the gradients of the individual losses
  std::vector<GradNonGradPair> opsToRegister = growLossGradients();

  while (!opsToRegister.empty()) {

    auto &toRegister = opsToRegister.back();
    registerOpGrads(toRegister.grad, toRegister.nongrad);
    opsToRegister.resize(opsToRegister.size() - 1);

    for (auto &nongrad_egrads : tensor_grad_registry.popComplete()) {

      Tensor *nongrad = getTensors().get(nongrad_egrads.first);
      const std::vector<Tensor *> &egrads = nongrad_egrads.second;
      // nongrad required below, as the name of the output of the
      // created op (sumOp) will be based off of it. Also, we
      // register the link between sumOp's output and nongrad
      Op *sumOp = growGradSumOp(nongrad, egrads);

      sumOp->fromLoss = PathFromLoss::Yes;
      logging::ir::trace("New gradient sumOp, {}, created", sumOp->str());

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
      case TensorType::Momentum:
      case TensorType::Unknown:
      case TensorType::N:
        throw error("can't currently register gradient of " +
                    nongrad->tensor_type() + " tensor, " + nongrad->str());

      default: {
        throw error("only handling ActGrad and Variable for now");
      }
      }
    }

    for (Op *op : op_grad_registry.popComplete()) {
      for (auto &gradOp : growGradOps(op)) {
        opsToRegister.push_back({gradOp, op});
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
    };
  }

  logging::ir::info("Constructing backwards complete");
}

Op *Ir::growCopyVarUpdateOp(const TensorId &varId, const TensorId &from) {
  OpId opId = getMainGraph().moveIntoGraph(
      std::unique_ptr<Op>(new CopyVarUpdateOp(varId, {getMainGraph(), ""})));

  // The order of inputs is important
  std::vector<TensorId> inputs{varId, from};
  getMainGraph().connectInputs(InputVecWrapper(inputs), opId);

  return growVarUpdateOpInternal(opId);
}

Op *Ir::growGradientVarUpdateOp(const TensorId &varId) {

  // A sanity check that the Tensor is not fixed point type
  if (getTensors().get(varId)->info.getDataTypeInfo()->isFixedPoint()) {
    throw error("Currently only floating point variable tensors are updatable");
  }
  OpId opId =
      getMainGraph().moveIntoGraph(optimizer->createOp(varId, getMainGraph()));
  auto inputs =
      optimizer->getInputIds(varId, getTensors().get(varId)->info.dataType());
  getMainGraph().connectInputs(InputVecWrapper(inputs), opId);
  return growVarUpdateOpInternal(opId);
}

Op *Ir::growVarUpdateOpInternal(OpId opId) {

  Op *op = getMainGraph().getOps()[opId].get();

  if (getSessionOptions().enableVirtualGraphs) {

    // Count which vgraph's the input's producer ops are on.
    std::map<int64_t, int64_t> vgraphIdMap;
    for (auto inputT : op->input->tensors()) {
      Op *producer = inputT->getProducerUnsafe();
      if (producer != nullptr) {
        boost::optional<int64_t> vgraphId = producer->getVirtualGraphId();
        if (vgraphId) {
          vgraphIdMap[*vgraphId]++;
        }
      }
    }
    // Find the vgraph id with the most occurrences.
    auto it = std::max_element(vgraphIdMap.begin(),
                               vgraphIdMap.end(),
                               [](const std::pair<int64_t, int64_t> &p1,
                                  const std::pair<int64_t, int64_t> &p2) {
                                 return p1.second < p2.second;
                               });

    op->setVirtualGraphId(it->first);
  }

  auto varUpdateOp = dynamic_cast<VarUpdateOp *>(op);
  if (varUpdateOp == nullptr) {
    throw error("Internal Logic Error (ILE) Op {} expected to be a VarUpdateOp",
                op->str());
  }
  TensorId updatedVarId = getUpdatedVarId(varUpdateOp->getVarId());
  std::vector<TensorId> outputs{updatedVarId};
  getMainGraph().connectOutputs(OutputVecWrapper(outputs), opId);
  op->setup();

  trainTargetOps.insert(op);
  return op;
}

void Ir::growFinalLoss() {
  if (losses.size() == 0) {
    throw error("In Ir::growFinalLoss, but losses vector is empty");
  }

  logging::ir::info("growing final loss");

  std::vector<Op *> lossOps;
  // first, grow each of the individual losses from the user
  for (auto &loss : losses) {
    OpId opId = getMainGraph().moveIntoGraph(loss->getOp({getMainGraph(), ""}));
    Op *lossOp = getMainGraph().getOps()[opId].get();
    logging::ir::trace("Connecting inputs/outputs for Loss Op {}",
                       lossOp->str());
    getMainGraph().connectInputs(*loss, opId);
    getMainGraph().connectOutputs(*loss, opId);
    lossOps.push_back(lossOp);
    lossOp->setup();
    lossOp->toLoss = PathToLoss::Yes;
    // there is no path from the final loss to this pre-final loss op
    lossOp->fromLoss = PathFromLoss::No;
  }

  // now growing the FINAL loss (sum of individual losses)
  std::unique_ptr<poponnx::Op> finalLossSum =
      OpManager::createOp(Domain::ai_onnx,
                          "Sum",
                          getOpSetVersionFromModel(Domain::ai_onnx),
                          getMainGraph(),
                          "FinalLoss");

  // The final Loss Op is the only Op which (we say) has both paths to and from
  finalLossSum->toLoss   = PathToLoss::Yes;
  finalLossSum->fromLoss = PathFromLoss::Yes;

  if (getSessionOptions().enableVirtualGraphs) {

    // Count which vgraph's the producer ops are on.
    std::map<int64_t, int64_t> vgraphIdMap;
    for (auto &l : lossOps) {
      boost::optional<int64_t> vgraphId = l->getVirtualGraphId();
      if (vgraphId) {
        vgraphIdMap[*vgraphId]++;
      }
    }

    // Find the vgraph id with the most occurrences.
    auto it = std::max_element(vgraphIdMap.begin(),
                               vgraphIdMap.end(),
                               [](const std::pair<int64_t, int64_t> &p1,
                                  const std::pair<int64_t, int64_t> &p2) {
                                 return p1.second < p2.second;
                               });

    finalLossSum->setVirtualGraphId(it->first);
  }

  finalLossOpId = getMainGraph().moveIntoGraph(std::move(finalLossSum));

  std::vector<TensorId> inputs;
  inputs.reserve(lossOps.size());
  for (auto &op : lossOps) {
    // Assume that tensor(0) is always valid
    inputs.push_back(op->output->tensor(0)->id);
  }
  std::vector<TensorId> outputs{getFinalLossId()};
  getMainGraph().connectInputs(InputVecWrapper(inputs), finalLossOpId);
  getMainGraph().connectOutputs(OutputVecWrapper(outputs), finalLossOpId);
  getMainGraph().getOps()[finalLossOpId]->setup();

  // Not necessary to set the phase here (it will be done in
  // updateVertices). To check our logic though, we do this here
  // and then check that we agree in updateVertices()
  logging::ir::trace("Final loss Op id set to {}", finalLossOpId);
}

TensorId Ir::getFinalLossId() const { return "finalLoss"; }

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

std::vector<GradNonGradPair> Ir::growLossGradients() {
  auto finalLossOpFound = getMainGraph().getOps().find(finalLossOpId);
  if (finalLossOpFound != getMainGraph().getOps().end()) {
    std::vector<GradNonGradPair> pairs;
    for (auto &t_inds : finalLossOpFound->second->input->indicesMap()) {
      Tensor *t = t_inds.first;
      // a Loss Op going into the final Sum
      Op *lossOp = t->getProducer();
      for (Op *gradOp : growGradOps(lossOp)) {
        pairs.push_back({gradOp, lossOp});
      }
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
  return getExecutionMode() == ExecutionMode::INFERENCE || canEvaluate();
}

bool Ir::canEvaluate() const {
  return getExecutionMode() == ExecutionMode::EVALUATION || canTrain();
}

bool Ir::canTrain() const {
  return getExecutionMode() == ExecutionMode::TRAINING;
}

bool Ir::containsInitialisers() {
  return !(onnxModel->graph().initializer().empty());
}

void Ir::applyUpdateInplacePrioritiesForIpu() {
  UpdateInplacePrioritiesForIpu pattern;

  for (auto &id_graph : graphs) {
    auto graph = id_graph.second.get();
    for (auto &id_op : graph->getOps()) {
      Op *op = id_op.second.get();
      pattern.apply(op);
    }
  }
}

void Ir::applyInplacePattern(Graph &graph) {

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

  auto tripletComparitor = [](const Triplet &a, const Triplet &b) {
    if (std::get<2>(a) - std::get<2>(b) != 0.0f) {
      return std::get<2>(a) > std::get<2>(b);
    }
    // if same priority, fall back to ID to keep it deterministic
    return std::get<0>(a) > std::get<0>(b);
  };

  if (priorities.size() != 0) {

    // sort in decreasing order of priority,
    std::sort(priorities.begin(), priorities.end(), tripletComparitor);

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
        priorities.begin(), priorities.end(), zeroPriority, tripletComparitor);
    priorities.erase(found, priorities.end());

    // we keep track of which ops have already been inplaced
    std::set<OpId> inplacedAlready;

    for (auto &ip : priorities) {
      OpId id                       = std::get<0>(ip);
      OperatorIdentifier identifier = std::get<1>(ip);
      // first check that the op has not already been inplaced:
      auto inplaced_already_it = inplacedAlready.find(id);
      if (inplaced_already_it != inplacedAlready.end()) {
        // the Op has already been inplaced
      } else {
        Op *op              = graph.getOps().at(id).get();
        bool touchesAnchors = false;
        for (auto &tensor : inplace.touches(op, identifier)) {
          if (isAnchored(tensor->id)) {
            touchesAnchors = true;
          }
        }

        // If it is recompute and uses inplace output, do not inplace.
        // This is conservative (aliasing can sometimes still be inplaced)
        // TODO T9352: use logic based on existing Inplace code
        // It can be shown that checkpoints consuming recomputable outputs
        // do not need to be inplaced
        bool recomputeUsingCheckpoint = false;
        if (op->settings.recomputeType == RecomputeType::RECOMPUTE) {
          for (auto &index_tensor : op->input->tensorMap()) {
            auto inTensor = index_tensor.second;
            if (!inTensor->hasProducer() ||
                (inTensor->hasProducer() &&
                 inTensor->getProducer()->settings.recomputeType ==
                     RecomputeType::CHECKPOINT)) {
              recomputeUsingCheckpoint = true;
            }
          }
        }

        if (!touchesAnchors && !recomputeUsingCheckpoint) {
          auto newTopoCons = inplace.getNewTopoCons(op, identifier);
          if (isSchedulable(newTopoCons)) {
            inplacedAlready.insert(op->id);
            inplace.apply(op, identifier, newTopoCons);
          }
        }
      }
    }
  }
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

  throw error("no tensor with id " + tensor_id);
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

bool Ir::hasGraph(const GraphId &graphId) const {
  return graphs.find(graphId) != graphs.end();
}

Graph &Ir::createGraph(const GraphId &graphId) {
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

Tensors &Ir::getMainGraphTensors() { return getMainGraph().getTensors(); }

const Tensors &Ir::getMainGraphTensors() const {
  return getMainGraph().getTensors();
}

uint32_t Ir::getAndIncrementDropoutSeedModifier() {
  dropoutSeedModifier += 1;
  return dropoutSeedModifier;
}

} // namespace poponnx
