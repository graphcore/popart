// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <queue>
#include <popart/aliasesmap.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/mainloops.hpp>
#include <popart/transforms/prune.hpp>

namespace popart {

namespace {

class OpsToMove {
public:
  OpsToMove(Graph &graph_, std::vector<Op *> ops_, Aliases &aliases_)
      : graph(graph_), ops(ops_), aliases(aliases_) {
    std::set<OpId> opIdSet;
    // Inputs
    for (Op *op : ops) {
      opIdSet.insert(op->id);
      for (auto input : op->input->tensorMap()) {
        if (!input.second->hasProducer() ||
            opIdSet.find(input.second->getProducer()->id) == opIdSet.end()) {
          addInput(input.second->id);
        }
      }
    }

    // Outputs
    for (Op *op : ops) {
      opIdSet.insert(op->id);
      for (auto output : op->output->tensorMap()) {
        if (output.second->isGraphOutput()) {
          addOutput(output.second->id);
        }
        for (Op *consumer : output.second->consumers.getOps()) {
          if (opIdSet.find(consumer->id) == opIdSet.end()) {
            addOutput(output.second->id);
          }
        }
      }
    }
  }

  void addInput(TensorId id) {
    if (inTensorIdxMap.find(id) == inTensorIdxMap.end()) {
      auto index            = inTensorIdxMap.size();
      inTensorIdxMap[id]    = index;
      inIdxTensorMap[index] = id;
    }
  }

  void addOutput(TensorId id) {
    if (outTensorIdxMap.find(id) == outTensorIdxMap.end()) {
      auto index             = outTensorIdxMap.size();
      outTensorIdxMap[id]    = index;
      outIdxTensorMap[index] = id;
    }
  }

  const std::vector<Op *> &getOps() const { return ops; }

  void evaluate() {
    std::set<TensorId> processed;

    opIds.clear();
    explicitTensorIds.clear();
    implicitTensorIds.clear();
    aliasMap.clear();
    modifiesMap.clear();

    for (Op *op : ops) {
      opIds.insert(op->id);
    }

    for (auto &out : outIdxTensorMap) {
      TensorId outTensorId = out.second;
      Tensor *outTensor    = graph.getTensors().get(outTensorId);

      auto aliasedTensorMap = aliases.aliasChainsFrom(outTensor);
      auto fullRegion       = view::Region::getFull(outTensor->info.shape());

      bool handled = false;
      for (auto &chain : aliasedTensorMap) {
        TensorId aliasedTensorId = chain.first->id;
        Tensor *aliasedTensor    = graph.getTensors().get(aliasedTensorId);

        auto regions = chain.second.apply(fullRegion);
        if (inTensorIdxMap.find(aliasedTensorId) != inTensorIdxMap.end() &&
            std::any_of(regions.begin(),
                        regions.end(),
                        [&chain](const view::Region &region) {
                          return view::Region::getFull(
                                     chain.first->info.shape()) == region;
                        })) {
          // Input ID -> Output ID
          auto inTensor       = aliasedTensor;
          TensorId inTensorId = inTensor->id;
          explicitTensorIds.push_back({inTensor->id, outTensor->id});
          auto fwdAliasRegions = aliases.getChainsFromTo(inTensor, outTensor);
          auto bwdAliasRegions = aliases.getChainsFromTo(outTensor, inTensor);

          aliasMap.insert(
              {{inTensorId, outTensorId}, {fwdAliasRegions, bwdAliasRegions}});

          processed.insert(chain.first->id);
          processed.insert(outTensorId);
          handled = true;
          break;
        }
      }
      if (!handled) {
        throw error("[MainLoops] Output {} cannot be loop carried.",
                    outTensorId);
      }
    }

    for (auto &in : inIdxTensorMap) {
      if (processed.find(in.second) == processed.end()) {
        implicitTensorIds.push_back(in.second);
        processed.insert(in.second);
      }
    }

    for (auto &in : inIdxTensorMap) {
      auto modifiedRegions =
          graph.getTensors().get(in.second)->modifiedRegionsByOps(ops, aliases);
      modifiesMap[in.second] = modifiedRegions;
    }

    evaluated = true;
  }

  const std::vector<std::pair<TensorId, TensorId>> &
  getExplicitTensorIds() const {
    checkEvaluated();
    return explicitTensorIds;
  }

  const std::vector<TensorId> &getImplicitTensors() const {
    checkEvaluated();
    return implicitTensorIds;
  }

  const std::map<std::pair<TensorId, TensorId>,
                 std::pair<view::Chains, view::Chains>> &
  getAliasMap() const {
    checkEvaluated();
    return aliasMap;
  }

  const std::map<TensorId, view::Regions> &getModifiesMap() const {
    checkEvaluated();
    return modifiesMap;
  }

  bool isEvaluated() { return evaluated; }

private:
  void checkEvaluated() const {
    if (!evaluated) {
      throw internal_error("[ExternalIO] evaluate() not called.");
    }
  }

  const Graph &graph;
  std::vector<Op *> ops;
  std::set<OpId> opIds;

  // TODO T40062: Replace use of chain-based aliasing.
  Aliases &aliases;

  // Flag to check if the
  bool evaluated;

  std::map<TensorId, int> inTensorIdxMap;
  std::map<TensorId, int> outTensorIdxMap;
  std::map<int, TensorId> inIdxTensorMap;
  std::map<int, TensorId> outIdxTensorMap;

  std::vector<std::pair<TensorId, TensorId>> explicitTensorIds;
  std::vector<TensorId> implicitTensorIds;

  std::map<std::pair<TensorId, TensorId>, std::pair<view::Chains, view::Chains>>
      aliasMap;
  std::map<TensorId, view::Regions> modifiesMap;
};

void finalizeSubgraphSettings(Graph &subgraph) {
  for (auto &op : subgraph.getOps()) {
    op.second->settings.executionContext = ExecutionContext::Subgraph;
  }
}

void moveIntoLoop(LoopOp *loop,
                  const OpsToMove &opsToMove,
                  std::map<TensorId, TensorId> &remap) {
  Graph &graph    = loop->getGraph();
  Ir &ir          = graph.getIr();
  Graph &subgraph = loop->getCalledGraph();

  logging::transform::trace(
      "[moveIntoLoop] Moving {} ops from {} to {} on LoopOp {}",
      opsToMove.getOps().size(),
      graph.id.str(),
      subgraph.id.str(),
      loop->debugName());

  // Add explicit inputs
  for (auto &explicitIds : opsToMove.getExplicitTensorIds()) {
    TensorId opInId = explicitIds.first;
    TensorId sgInId = subgraph.addScope(graph.removeScope(opInId));
    logging::transform::trace(
        "[moveIntoLoop] Adding explicit input {} -> {} on LoopOp {}",
        opInId,
        sgInId,
        loop->debugName());
    loop->addLoopInput(
        std::max(LoopOp::getFirstInputInIndex(), loop->input->maxIndex() + 1),
        opInId,
        sgInId,
        false);
    remap[opInId] = sgInId;
  }

  // Add implicit inputs
  for (auto &implicitId : opsToMove.getImplicitTensors()) {
    TensorId opInId = implicitId;
    TensorId sgInId = subgraph.addScope(graph.removeScope(opInId));
    logging::transform::trace(
        "[moveIntoLoop] Adding implicit input {} -> {} on LoopOp {}",
        opInId,
        sgInId,
        loop->debugName());
    loop->addLoopInput(
        std::max(LoopOp::getFirstInputInIndex(), loop->input->maxIndex() + 1),
        opInId,
        sgInId,
        false);
    remap[opInId] = sgInId;
  }

  // Map from old (existing) Op -> new (replacement) Op(s)
  std::map<Op *, std::vector<Op *>> opRemaps;

  for (Op *op : opsToMove.getOps()) {
    logging::transform::trace("[moveIntoLoop] Moving Op {} from {} to {}",
                              op->debugName(),
                              graph.id.str(),
                              subgraph.id.str());

    auto cloneOpUp = op->clone();
    Op *cloneOp    = cloneOpUp.get();
    opRemaps.insert({op, {cloneOp}});

    subgraph.moveIntoGraph(std::move(cloneOpUp));

    cloneOp->setScope(subgraph.getScope());

    auto connectInTensorFn =
        [](Op *srcOpP, Op *dstOpP, InIndex index, TensorId tensorId) {
          IpuCopyOp *srcOp = dynamic_cast<IpuCopyOp *>(srcOpP);
          IpuCopyOp *dstOp = dynamic_cast<IpuCopyOp *>(dstOpP);
          if (srcOp && dstOp) {
            TensorId srcTensorId = srcOp->input->tensor(index)->id;
            dstOp->connectInTensor(
                index, tensorId, srcOp->getSourceIpu(srcTensorId));
          } else {
            dstOpP->connectInTensor(index, tensorId);
          }
        };

    for (auto input : op->input->tensorIdMap()) {
      TensorId subgraphTensorId;
      auto find = remap.find(input.second);
      if (find == remap.end()) {
        throw internal_error("[moveIntoLoop] Failed to get input {}",
                             input.second);
      }
      subgraphTensorId = find->second;
      connectInTensorFn(op, cloneOp, input.first, subgraphTensorId);
    }
    for (auto output : op->output->tensorIdMap()) {
      TensorId subgraphTensorId =
          subgraph.addScope(graph.removeScope(output.second));
      cloneOp->createAndConnectOutTensor(output.first, subgraphTensorId);
      remap[output.second] = subgraphTensorId;
    }
    cloneOp->setup();
  }

  // Transfer topocons
  TopoCons::transferToSubgraph(loop, opRemaps);

  // Disconnect and delete old ops
  for (Op *op : opsToMove.getOps()) {
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
    op->getGraph().eraseOp(op->id);
  }

  // Add outputs
  for (auto &explicitIds : opsToMove.getExplicitTensorIds()) {
    TensorId opOutId = explicitIds.second;
    TensorId sgOutId = remap[opOutId];
    logging::transform::trace(
        "[moveIntoLoop] Adding explicit output {} -> {} on LoopOp {}",
        sgOutId,
        opOutId,
        loop->debugName());
    loop->addLoopOutput(loop->output->maxIndex() + 1, opOutId, sgOutId, false);
  }

  // Transfer input -> output aliases
  for (auto &aliased : opsToMove.getAliasMap()) {
    TensorId opInId  = aliased.first.first;
    TensorId opOutId = aliased.first.second;
    logging::transform::trace(
        "[moveIntoLoop] Adding alias input {} -> output {} on LoopOp {}",
        opInId,
        opOutId,
        loop->debugName());
    for (auto &inIndex : loop->input->indices(graph.getTensors().get(opInId))) {
      for (auto &outIndex :
           loop->output->indices(graph.getTensors().get(opOutId))) {
        loop->addAlias(
            inIndex, outIndex, aliased.second.first, aliased.second.second);
      }
    }
  }

  // Transfer modified inputs
  for (auto &modified : opsToMove.getModifiesMap()) {
    TensorId opInId = modified.first;
    logging::transform::trace(
        "[moveIntoLoop] Adding modified input {} on LoopOp {}",
        opInId,
        loop->debugName());
    for (auto &inIndex :
         loop->input->indices(graph.getTensors().get(modified.first))) {
      logging::transform::trace(
          "[moveIntoLoop] Adding modified input {}: {} on LoopOp {}",
          inIndex,
          modified.first,
          loop->debugName());
      loop->addModified(inIndex, modified.second);
    }
  }

  for (auto tIds : remap) {
    if (ir.isAnchored(tIds.first)) {
      logging::transform::trace(
          "[moveIntoLoop] Remapping anchor {} -> {}", tIds.first, tIds.second);
      ir.remapAnchor(tIds.first, tIds.second);
    }
  }

  auto sgvgraphs = subgraph.getAllVirtualGraphIds(false);
  if (!sgvgraphs.empty()) {
    logging::transform::trace("[moveIntoLoop] Setting {} to VGID: {}",
                              loop->debugName(),
                              *sgvgraphs.begin());
    loop->setVirtualGraphId(*sgvgraphs.begin());
  }

  // Remove moved tensors and ops
  Transform::applyTransform(Prune::id(), graph);
}

} // namespace

std::size_t MainLoops::id() { return typeid(MainLoops).hash_code(); }

Graph &MainLoops::getInnerLoopSubgraph(const Ir &ir) {
  if (ir.getSessionOptions().getAccumulationFactor() > 1) {
    return ir.getGraph(GraphId(getAccumulationGraphName()));
  } else {
    return ir.getGraph(GraphId(getStepGraphName()));
  }
}

LoopOp *MainLoops::getInnerLoopOp(const Ir &ir) {
  auto getLoopCallsiteOpsOfSubgraph = [](const Ir &ir, const Graph &graph) {
    std::vector<LoopOp *> callSites;
    for (auto op : ir.getAllOps()) {
      auto loopOp = dynamic_cast<LoopOp *>(op);
      if (loopOp) {
        callSites.push_back(loopOp);
      }
    }
    return callSites;
  };

  auto loopOps = getLoopCallsiteOpsOfSubgraph(ir, getInnerLoopSubgraph(ir));

  if (loopOps.size() > 1 || loopOps.size() == 0) {
    throw error(
        "[MainLoops] Unable to find inner loop op. {} candidates found.",
        loopOps.size());
  } else {
    return loopOps.at(0);
  }
}

bool MainLoops::apply(Graph &graph) const {
  auto &ir                = graph.getIr();
  auto &sessionOptions    = ir.getSessionOptions();
  auto accumulationFactor = sessionOptions.getAccumulationFactor();
  auto batchesPerStep     = ir.getDataFlow().batchesPerStep();

  // Check whether the anchor return types are supported.
  for (auto &anchorArt : ir.getDataFlow().getAnchorReturnTypeMap()) {
    // TODO(T39577): Uncomment the AnchorReturnTypeId::Sum line in the if
    // statement below once this is resolved, and fix the last sentence in the
    // error message to "Supported anchor return types are
    // AnchorReturnTypeId::All and AnchorReturnTypeId::Sum."
    if (!(anchorArt.second.id() == AnchorReturnTypeId::All
          // || anchorArt.second.id() == AnchorReturnTypeId::Sum
          )) {
      throw error(
          "AnchorReturnType::{} for TensorId \"{}\" is unsupported when "
          "explicit main loops are enabled. Supported anchor return type "
          "is AnchorReturnTypeId::All.", // and AnchorReturnTypeId::Sum.",
          anchorArt.second.str(),
          anchorArt.first);
    }
  }

  // Initially: Point all IDs to the existing main graph
  GraphId mainGraphId  = graph.id;
  GraphId stepGraphId  = graph.id;
  GraphId accumGraphId = graph.id;

  // Optional loops
  LoopOp *stepLoop  = nullptr;
  LoopOp *accumLoop = nullptr;
  std::map<TensorId, TensorId> stepTensorRemap;
  std::map<TensorId, TensorId> accumTensorRemap;
  std::map<OpId, OpId> stepOpRemap;
  std::map<OpId, OpId> accumOpRemap;

  if (batchesPerStep > 1) {
    Op::Settings stepLoopSettings(graph, "stepLoop");
    stepLoopSettings.executionContext = ExecutionContext::Normal;

    Graph &stepGraph = ir.createGraph({"stepGraph"});

    // Add mandatory loop iterator tensor to subgraph (is not an output)
    TensorId loopItScopedId = stepGraph.addScope(reservedLoopIteratorPrefix());
    stepGraph.addInput(loopItScopedId, TensorInfo(DataType::INT32, {}));

    // Add mandatory loop condition tensor to subgraph (is also an output)
    TensorId loopCondScopedId = stepGraph.addScope(reservedLoopCondPrefix());
    stepGraph.addInput(loopCondScopedId, TensorInfo(DataType::BOOL, {}));
    stepGraph.markAsOutput(loopCondScopedId);

    stepGraphId  = stepGraph.id;
    accumGraphId = stepGraph.id;

    stepLoop = graph.createOp<LoopOp>(
        Onnx::Operators::Loop_11, stepLoopSettings, stepGraph);
    logging::transform::trace("[MainLoops] step loop trip count: {}",
                              batchesPerStep);
    stepLoop->setTripCountValue(batchesPerStep);
  }

  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);

  // Move operations into the step loop
  if (batchesPerStep > 1) {
    std::vector<Op *> ops;
    std::set<TensorId> requiredOutTensorIds;
    for (Op *op : schedule) {
      if ((op->settings.executionContext == ExecutionContext::Normal ||
           op->settings.executionContext ==
               ExecutionContext::AccumulateOuterFragment) &&
          op->id != stepLoop->id) {
        logging::transform::trace("[MainLoops] Moving {} into step loop",
                                  op->debugName());
        ops.push_back(op);
      }
    }
    AliasesMap aliasesMap{graph};
    Aliases &aliases = aliasesMap.getAliases(graph.id);
    OpsToMove opsToMove(graph, ops, aliases);
    opsToMove.evaluate();
    moveIntoLoop(stepLoop, opsToMove, stepTensorRemap);
  }
  ir.updateVertices();

  schedule =
      ir.getGraph(stepGraphId).getOpSchedule({}, RequireOptimalSchedule::No);

  if (accumulationFactor > 1) {
    Graph &stepGraph = ir.getGraph(stepGraphId);

    Op::Settings accumLoopSettings(stepGraph, "accumulationLoop");
    if (stepLoop) {
      accumLoopSettings.executionContext = ExecutionContext::Subgraph;
    } else {
      accumLoopSettings.executionContext = ExecutionContext::Normal;
    }

    Graph &accumGraph = ir.createGraph({getAccumulationGraphName()});

    // Add mandatory loop iterator tensor to subgraph (is not an output)
    TensorId loopItScopedId = accumGraph.addScope(reservedLoopIteratorPrefix());
    accumGraph.addInput(loopItScopedId, TensorInfo(DataType::INT32, {}));

    // Add mandatory loop condition tensor to subgraph (is also an output)
    TensorId loopCondScopedId = accumGraph.addScope(reservedLoopCondPrefix());
    accumGraph.addInput(loopCondScopedId, TensorInfo(DataType::BOOL, {}));
    accumGraph.markAsOutput(loopCondScopedId);

    accumGraphId = accumGraph.id;

    accumLoop = stepGraph.createOp<LoopOp>(
        Onnx::Operators::Loop_11, accumLoopSettings, accumGraph);
    logging::transform::trace("[MainLoops] accumulation loop trip count: {}",
                              accumulationFactor);
    accumLoop->setTripCountValue(accumulationFactor);
  }

  // Move operations into the accumulation loop
  if (accumulationFactor > 1) {
    std::vector<Op *> ops;
    std::set<TensorId> requiredOutTensorIds;
    for (Op *op : schedule) {
      if (op->settings.executionContext == ExecutionContext::Normal &&
          op->id != accumLoop->id) {
        logging::transform::trace(
            "[MainLoops] Moving {} into accumulation loop", op->debugName());
        ops.push_back(op);
      }
    }
    auto &stepGraph = ir.getGraph(stepGraphId);
    AliasesMap aliasesMap{stepGraph};
    auto &stepGraphAliases = aliasesMap.getAliases(stepGraphId);
    OpsToMove opsToMove(stepGraph, ops, stepGraphAliases);
    opsToMove.evaluate();
    moveIntoLoop(accumLoop, opsToMove, accumTensorRemap);
  }
  ir.updateVertices();

  // Set relevant Op settings
  if (stepGraphId != graph.id) {
    auto &stepGraph = ir.getGraph(stepGraphId);
    finalizeSubgraphSettings(stepGraph);
  }

  if (accumGraphId != graph.id) {
    auto &accumGraph = ir.getGraph(accumGraphId);
    finalizeSubgraphSettings(accumGraph);
  }

  // Remove accumulation outer fragment contexts
  schedule = graph.getOpSchedule({}, RequireOptimalSchedule::No);
  for (Op *op : schedule) {
    if (op->settings.executionContext ==
        ExecutionContext::AccumulateOuterFragment) {
      op->settings.executionContext = ExecutionContext::Normal;
    }
  }

  if (stepLoop != nullptr) {
    logging::transform::debug("[MainLoops] step loop: {}",
                              stepLoop->debugName());
  }
  if (accumLoop != nullptr) {
    logging::transform::debug("[MainLoops] accumulation loop: {}",
                              accumLoop->debugName());
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new MainLoops);
}

} // namespace popart
