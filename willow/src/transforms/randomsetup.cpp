// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/modifyrandomseed.hpp>
#include <popart/op/randombase.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/randomsetup.hpp>

#include "popart/tensornames.hpp"
#include "popart/vendored/optional.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

namespace popart {

TensorInfo RandomSetup::seedTensorInfo(DataType::UINT32, {2});

std::size_t RandomSetup::id() { return typeid(RandomSetup).hash_code(); }

bool RandomSetup::apply(Graph &graph) const {

  auto &ir = graph.getIr();

  if (RandomSetup::requiresRandomSeed(ir)) {

    logging::debug("[RandomSetup] Started.");

    // Determine what we are going to do.
    auto cfg = determineConfig(ir);

    // Log what we are going to do.
    logConfig(ir, cfg);

    // Do the actual transformation step-wise.
    for (auto &graphId : cfg.graphApplyOrder) {
      applyToGraph(ir.getGraph(graphId), cfg);
    }

    logging::debug("[RandomSetup] Done.");
    return true;

  } else {
    logging::debug("[RandomSetup] Nothing to do.");
  }

  return false;
}

RandomSetup::Config RandomSetup::determineConfig(const Ir &ir) const {
  Config cfg;

  // Work out what attributes we need to set for new ops.
  cfg.setVirtualGraphIds = determineSetVirtualGraphIds(ir);
  cfg.setExecutionPhases = determineSetExecutionPhases(ir);
  cfg.setPipelineStages  = determineSetPipelineStages(ir);

  // Get op strand mapping and random ops mapping.
  auto res         = determineStrandsMapAndRandomOpsMap(ir);
  cfg.strandsMap   = std::get<0>(res);
  cfg.randomOpsMap = std::get<1>(res);

  // Get TensorIds for random seeds.
  auto baseSeedMaps  = determineBaseSeedsMaps(ir, cfg.strandsMap);
  cfg.inBaseSeedMap  = std::get<0>(baseSeedMaps);
  cfg.outBaseSeedMap = std::get<1>(baseSeedMaps);

  // Determine where to insert inputs/outputs.
  auto seedIndexMaps       = determineSeedIndexMaps(ir);
  cfg.firstSeedInIndexMap  = std::get<0>(seedIndexMaps);
  cfg.firstSeedOutIndexMap = std::get<1>(seedIndexMaps);

  // Determine order of application.
  cfg.graphApplyOrder = determineGraphApplyOrder(ir);

  return cfg;
}

bool RandomSetup::determineSetVirtualGraphIds(const Ir &ir) const {
  return ir.virtualGraphsEnabled();
}

bool RandomSetup::determineSetExecutionPhases(const Ir &ir) const {
  return ir.getSessionOptions().executionPhaseSettings.phases > 1;
}

bool RandomSetup::determineSetPipelineStages(const Ir &ir) const {
  auto &opts = ir.getSessionOptions();
  auto ops   = ir.getAllOps();

  auto opPred = [](Op *op) { return op->hasPipelineStage(); };

  return opts.enablePipelining && std::all_of(ops.begin(), ops.end(), opPred);
}

RandomSetup::StrandsMapAndRandomOpsMap
RandomSetup::determineStrandsMapAndRandomOpsMap(const Ir &ir) const {

  auto graphs = ir.getAllGraphs();

  GraphToStrands strandsMap;
  GraphToOpToStrands randomOpsMap;

  // Initially populate randomOpsMap with all ops that are derived from
  // RandomBaseOp. Note that there are also 'subgraph ops' that will need to be
  // added later, but that's done in the while loop below.
  std::transform(graphs.begin(),
                 graphs.end(),
                 std::inserter(randomOpsMap, randomOpsMap.end()),
                 [&](const Graph *graph) -> GraphToOpToStrands::value_type {
                   OpToStrands randomOps = getInitStrandToOps(*graph);
                   return {graph->id, randomOps};
                 });

  // Initially populate 'strandsMap' with those strands that have an
  // op attached to them in 'randomOpsMap'.
  std::transform(graphs.begin(),
                 graphs.end(),
                 std::inserter(strandsMap, strandsMap.end()),
                 [&](const Graph *graph) -> GraphToStrands::value_type {
                   // Get keys from randomOpsMap.
                   Strands strands;
                   for (auto opIt = randomOpsMap.at(graph->id).begin();
                        opIt != randomOpsMap.at(graph->id).end();
                        ++opIt) {
                     for (const auto &strand : opIt->second) {
                       if (std::find(strands.begin(), strands.end(), strand) ==
                           strands.end()) {
                         strands.push_back(strand);
                       }
                     }
                   }
                   return {graph->id, strands};
                 });

  // Do fixed point calculation to properly populate both maps (e.g.
  // to work out which graphs need a seed). Note that we need a fixed point
  // because seeds requirements cascade through the IR -- that is, even if a
  // subgraph itself has no random ops it may still require a seed because it
  // contains a CallOp to a child subgraph that requires a seed.

  while (true) {
    bool hasChanged = false;

    for (auto &graph : graphs) {
      // If any subgraph op exists to a subgraph that is random, then this
      // subgraph also needs a random seed, because we need to pass it to
      // the subgraph.
      auto &graphStrands   = strandsMap.at(graph->id);
      auto &graphRandomOps = randomOpsMap.at(graph->id);

      for (auto &x : graph->getOps()) {
        Op *op = x.second.get();

        for (const auto &calledGraph : op->getCalledGraphs()) {

          auto &strandsInCalledGraph = strandsMap.at(calledGraph->id);
          for (auto &calledStrand : strandsInCalledGraph) {

            auto graphStrandsIt = std::find(
                graphStrands.begin(), graphStrands.end(), calledStrand);

            if (graphStrandsIt == graphStrands.end()) {

              // Log it.
              logging::trace("[RandomSetup] Determined {} requires random seed "
                             "for strand {} because of Op {}.",
                             graph->getGraphString(),
                             calledStrand,
                             op->str());

              // A subgraph op calls a subgraph that needs a random seed for a
              // strand and we haven't marked this strand as needed in this
              // graph as yet. Record it now.
              graphStrands.push_back(calledStrand);

              // We haven't reached a fixed point yet.
              hasChanged = true;
            }

            // The calledStrand should be a member of graphRandomOps[op].
            auto &opStrands = graphRandomOps[op];
            auto opStrandsIt =
                std::find(opStrands.begin(), opStrands.end(), calledStrand);

            if (opStrandsIt == opStrands.end()) {
              opStrands.push_back(calledStrand);

              // Log it.
              logging::trace("[RandomSetup] Determined {} requires random seed "
                             "for strand {} because of called {} of Op {}.",
                             graph->getGraphString(),
                             calledStrand,
                             calledGraph->getGraphString(),
                             op->str());

              // We haven't reached a fixed point yet.
              hasChanged = true;
            }
          }
        }
      }
    }

    if (!hasChanged) {
      // Fixpoint!
      break;
    }
  }

  return {strandsMap, randomOpsMap};
}

RandomSetup::OpToStrands
RandomSetup::getInitStrandToOps(const Graph &graph) const {
  // Get an initial mapping from strands to a list of ops for a given graph,
  // just populating with those ops derived from RandomBaseOp for now.
  OpToStrands randomOps;
  for (auto &x : graph.getOps()) {
    Op *op = x.second.get();
    if (auto randomOp = dynamic_cast<RandomBaseOp *>(op)) {
      if (!randomOp->hasInput(randomOp->getSeedInIndex())) {
        auto strand = getStrand(randomOp);
        randomOps[randomOp].push_back(strand);
        // Log it.
        logging::trace("[RandomSetup] Determined {} requires random seed "
                       "for strand {} because of Op {}.",
                       graph.getGraphString(),
                       strand,
                       randomOp->str());
      }
    }
  }
  return randomOps;
}

RandomSetup::InAndOutBaseSeedMap
RandomSetup::determineBaseSeedsMaps(const Ir &ir,
                                    const GraphToStrands &strandsMap) const {

  GraphToStrandToTensorId inBaseSeedIds;
  GraphToStrandToTensorId outBaseSeedIds;

  for (const auto &entry : strandsMap) {

    const auto &graphId   = entry.first;
    const auto &opStrands = entry.second;
    const auto &graph     = ir.getGraph(graphId);

    if (graphId == ir.getMainGraph().id) {
      // The main graph always uses the output of the GetRandomSeedOp, hence use
      // [randomSeed___updated] for for all strands.
      for (auto &strand : opStrands) {
        inBaseSeedIds[graphId][strand] =
            GetRandomSeedOp::getUpdatedSeedTensorId();
      }
      // No out base seeds needed for main graph. Outputs are only used on
      // subgraphs.
    } else {
      // Introduce new TensorIds specifically for this graph/strand of the form
      // <graph_id>
      for (auto &strand : opStrands) {
        auto id = ModifyRandomSeedOp::getSeedInTensorId();
        id      = getTensorIdForStrand(id, strand);
        inBaseSeedIds[graphId][strand]  = graph.addScope(id + "_in");
        outBaseSeedIds[graphId][strand] = graph.addScope(id + "_out");
      }
    }
  }

  return {inBaseSeedIds, outBaseSeedIds};
}

RandomSetup::FirstSeedInIndexMapAndFirstSeedOutIndexMap
RandomSetup::determineSeedIndexMaps(const Ir &ir) const {
  GraphToInIndex inIndices;
  GraphToOutIndex outIndices;

  const int loopFirstInput  = LoopOp::getFirstInputInIndex();
  const int loopFirstOutput = LoopOp::getFirstOutputOutIndex();

  for (const auto &graph : ir.getAllGraphs()) {
    // Work out a suitable place to put the input index for a base seed input
    // in a subgraph. Note that if subgraph is used as a loop body inputs are
    // offset by the loop iteration counter and the loop conditional inputs.
    // Additionally, any explicit input needs to have an associated output. We
    // want the seed to be an explicit input because it needs to be updated in
    // every loop iteration, so it should be added to the list of explicit
    // inputs. Our way of making sure this happens safely is by adding it
    // immediately after the loop condition input, if it exists. Note that
    // a subgraph need not be used in a loop, so we also need to deal with a
    // scenario where the number of inputs does not allow us to put the input
    // after the loop condition input.
    OutIndex loopBodyIn = loopFirstInput;
    // Cap input index by the number of inputs.
    loopBodyIn = std::min<InIndex>(loopBodyIn, graph->getInputIds().size());
    inIndices[graph->id] = loopBodyIn;

    // Work out a suitable place to put the base seed output. We need to make
    // sure that in subgraphs that are used as loop bodies these outputs are
    // loop carried with the base seed inputs. To do this, we need to correct
    // for the offset caused by the loop iteration and loop condition inputs
    // as well as the loop condition output. Bear in mind the subgraph may not
    // be a loop body -- this code needs to work for any subgraph.
    OutIndex loopBodyOut = loopBodyIn - loopFirstInput + loopFirstOutput;
    // Cap output index by the number of output.
    loopBodyOut = std::min<OutIndex>(loopBodyOut, graph->getOutputIds().size());
    // Index can't be less than 0.
    loopBodyOut = std::max<OutIndex>(loopBodyOut, 0);

    outIndices[graph->id] = loopBodyOut;
  }

  return {inIndices, outIndices};
}

RandomSetup::GraphIds
RandomSetup::determineGraphApplyOrder(const Ir &ir) const {
  const auto &graphs = ir.getAllGraphs();

  // We need called subgraphs to have been transformed before we transform
  // the ops that call them. To that end, we're restricted in the order in
  // which we can transform graphs. We determine that order in this function.

  GraphIds done;
  GraphIds todo;

  // Start with all graph ids in todo.
  std::transform(graphs.begin(),
                 graphs.end(),
                 std::back_inserter(todo),
                 [](auto &graph) { return graph->id; });

  // Checks if a graph is in done.
  auto calledGraphInDone = [&](const Graph *calledGraph) {
    auto doneIt = std::find(done.begin(), done.end(), calledGraph->id);
    return doneIt != done.end();
  };

  // Check if all called graphs are in done.
  using GetOpsEntry = std::pair<const OpId, std::unique_ptr<popart::Op>>;
  auto opsCalledGraphsInDone = [&](const GetOpsEntry &entry) {
    const auto &op      = entry.second;
    auto opCalledGraphs = op->getCalledGraphs();
    return std::all_of(
        opCalledGraphs.begin(), opCalledGraphs.end(), calledGraphInDone);
  };

  // While we still have graph ids in todo.
  while (!todo.empty()) {
    // Iterate over the todo list.
    for (auto todoIt = todo.begin(); todoIt != todo.end(); /* not here */) {
      const auto &graph        = ir.getGraph(*todoIt);
      bool allCalledGraphsDone = std::all_of(
          graph.getOps().begin(), graph.getOps().end(), opsCalledGraphsInDone);

      if (allCalledGraphsDone) {
        // Graph can be applied now, iterator to point to next element.
        done.push_back(*todoIt);
        todoIt = todo.erase(todoIt);
      } else {
        // Move on to next graph.
        ++todoIt;
      }
    }
  }

  return done;
}

void RandomSetup::logConfig(const Ir &ir, const Config &cfg) const {

  // This function just logs the 'cfg' object which defines all paramters
  // of the transformation we are about to do.

  // Log application order.
  std::vector<std::string> graphStrs;
  std::transform(cfg.graphApplyOrder.begin(),
                 cfg.graphApplyOrder.end(),
                 std::back_inserter(graphStrs),
                 [&](GraphId id) { return ir.getGraph(id).getGraphString(); });
  logging::trace("[RandomSetup] Determined [{}] is a valid application order.",
                 logging::join(graphStrs.begin(), graphStrs.end(), ", "));

  // Log each graph's params.
  for (const auto &graphId : cfg.graphApplyOrder) {
    auto &graph            = ir.getGraph(graphId);
    const Strands &strands = cfg.strandsMap.at(graphId);

    if (!strands.empty()) {
      // Create a string to log the strands for this graph.
      auto graphStrandsStr =
          logging::join(strands.begin(), strands.end(), ", ");
      logging::trace("[RandomSetup] Determined a random seed is required for "
                     "strand(s) {} in {} because:",
                     graphStrandsStr,
                     graph.getGraphString());

      for (auto entry : cfg.randomOpsMap.at(graphId)) {
        auto op       = entry.first;
        auto &strands = entry.second;

        auto opStrandsStr = logging::join(strands.begin(), strands.end(), ", ");
        logging::trace("[RandomSetup]   - op '{}' needs seed(s) for strand(s) "
                       "{}",
                       op->str(),
                       opStrandsStr);
      }
    } else {
      logging::trace("[RandomSetup] {} does not require a random seed.",
                     graph.getGraphString());
    }
  }
}

void RandomSetup::applyToGraph(Graph &graph, const Config &cfg) const {

  // Log what we are doing.
  logging::debug("[RandomSetup] Started transforming {}.",
                 graph.getGraphString());

  // Add base seeds for this graph as per the config.
  addBaseSeeds(graph, cfg);
  // Add seeds for operations in the graph.
  auto opSeeds = addModifyRandomSeedOps(graph, cfg);

  // Connect up seed tensors to all the ops in the graph.
  for (const auto &entry : cfg.randomOpsMap.at(graph.id)) {
    Op *op = entry.first;
    connectOp(graph, cfg, opSeeds.at(op), op);
  }

  // Log what we are doing.
  logging::debug("[RandomSetup] Done transforming {}.", graph.getGraphString());
}

void RandomSetup::addBaseSeeds(Graph &graph, const Config &cfg) const {
  bool isMainGraph = (graph.getIr().getMainGraph().id == graph.id);

  if (isMainGraph) {
    // The GetRandomSeedOp adds these base seeds.
    addGetRandomSeedOp(graph.getIr(), cfg);
  } else {
    // Add input tensors for base seeds.
    for (const auto &strand : cfg.strandsMap.at(graph.id)) {

      // Add graph input for strand.
      auto inBaseSeedId = cfg.inBaseSeedMap.at(graph.id).at(strand);
      auto inIndex      = cfg.firstSeedInIndexMap.at(graph.id);

      graph.addInput(inIndex, inBaseSeedId, seedTensorInfo, false);

      logging::trace("[RandomSetup] Added {} to {} as graph input #{} for "
                     "strand {}",
                     inBaseSeedId,
                     graph.getGraphString(),
                     inIndex,
                     strand);

      // Add graph output for strand.
      auto outBaseSeedId = cfg.outBaseSeedMap.at(graph.id).at(strand);
      auto outIndex      = cfg.firstSeedOutIndexMap.at(graph.id);

      graph.getTensors().addActGrad(outBaseSeedId, "seedOutput");
      auto tensor  = graph.getTensors().get(outBaseSeedId);
      tensor->info = seedTensorInfo;

      graph.markAsOutput(outIndex, outBaseSeedId, false);

      logging::trace("[RandomSetup] Added {} to {} as graph output #{} for "
                     "strand {}",
                     outBaseSeedId,
                     graph.getGraphString(),
                     outIndex,
                     strand);
    }
  }
}

void RandomSetup::addGetRandomSeedOp(Ir &ir, const Config &cfg) const {

  auto &graph = ir.getMainGraph();

  // 1. Create [randomSeed___fromHost] tensor.
  TensorId randomSeedFromHost = GetRandomSeedOp::getStreamedSeedTensorId();

  graph.getTensors().addStream(randomSeedFromHost, seedTensorInfo);
  Tensor &seedTensor = *graph.getTensors().get(randomSeedFromHost);
  seedTensor.setReplicatedStreamMode(Tensor::ReplicatedStreamMode::Replicate);

  logging::debug("[RandomSetup] Added tensor {}.", randomSeedFromHost);

  // 2. Create TensorId for [randomSeed___updated] (created later).
  TensorId randomSeedUpdated = GetRandomSeedOp::getUpdatedSeedTensorId();

  // 3. Create GetRandomSeedOp.
  Op::Settings settings(graph, "");
  auto getRandomSeedOp = std::make_unique<GetRandomSeedOp>(
      Onnx::CustomOperators::GetRandomSeed, settings);

  // Connect input.
  getRandomSeedOp->connectInTensor(getRandomSeedOp->getSeedInIndex(),
                                   randomSeedFromHost);
  // Create and connect output.
  getRandomSeedOp->createAndConnectOutTensor(
      GetRandomSeedOp::getUpdatedSeedOutIndex(), randomSeedUpdated);

  // Configure it.
  if (cfg.setExecutionPhases) {
    getRandomSeedOp->setExecutionPhase(0);
  }
  if (cfg.setVirtualGraphIds) {
    getRandomSeedOp->setVirtualGraphId(0);
  }
  if (cfg.setPipelineStages) {
    getRandomSeedOp->setPipelineStage(0);
  }

  // Call setup.
  getRandomSeedOp->setup();

  // Log it.
  logging::debug("[RandomSetup] Created op {} in {}.",
                 getRandomSeedOp->str(),
                 graph.getGraphString());

  // Add to graph.
  graph.moveIntoGraph(std::move(getRandomSeedOp));
}

TensorId
RandomSetup::addModifyRandomSeedOp(Graph &graph,
                                   const Config &cfg,
                                   const Strand &strand,
                                   uint32_t modifier,
                                   nonstd::optional<TensorId> opSeedId,
                                   const std::string &seedReasonStr) const {

  TensorId inBaseSeedId = cfg.inBaseSeedMap.at(graph.id).at(strand);

  auto constId = ModifyRandomSeedOp::getSeedModifierTensorId(modifier);
  constId      = getTensorIdForStrand(constId, strand);
  constId      = graph.addScope(constId);

  // Insert a constant tensor modifier for this op.
  std::vector<uint32_t> modifierData(1, {modifier});
  TensorInfo modifierInfo(DataType::UINT32, {});
  graph.getTensors().addConstInit(
      constId, modifierInfo, reinterpret_cast<void *>(modifierData.data()));

  auto &virtualGraphId = std::get<0>(strand);
  auto &pipelineStage  = std::get<1>(strand);

  Op::Settings settings(graph, "");
  auto modifyRandomSeedOp = std::make_unique<ModifyRandomSeedOp>(
      Onnx::CustomOperators::ModifyRandomSeed, settings);

  modifyRandomSeedOp->connectInTensor(modifyRandomSeedOp->getSeedInIndex(),
                                      inBaseSeedId);
  modifyRandomSeedOp->connectInTensor(
      modifyRandomSeedOp->getSeedModifierInIndex(), constId);

  if (!opSeedId) {

    // If opSeedId is not set, we're creating a new tensor to hold the seed
    // for the op.

    TensorId outId = ModifyRandomSeedOp::getModifiedSeedTensorId(modifier);
    outId          = getTensorIdForStrand(outId, strand);
    outId          = graph.addScope(outId);

    modifyRandomSeedOp->createAndConnectOutTensor(
        ModifyRandomSeedOp::getModifiedSeedOutIndex(), outId);

  } else {

    // If opSeedId is set, we're using an existing graph output.

    modifyRandomSeedOp->connectOutTensor(
        ModifyRandomSeedOp::getModifiedSeedOutIndex(), *opSeedId);
  }

  modifyRandomSeedOp->setup();

  // Configure it.
  if (cfg.setExecutionPhases) {
    modifyRandomSeedOp->setExecutionPhase(0);
  }
  if (cfg.setVirtualGraphIds) {
    modifyRandomSeedOp->setVirtualGraphId(virtualGraphId);
  }
  if (cfg.setPipelineStages) {
    modifyRandomSeedOp->setPipelineStage(pipelineStage);
  }

  logging::debug("[RandomSetup] Added op {} to {} in strand {} to provide "
                 "seeds for {}. ",
                 modifyRandomSeedOp->debugName(),
                 graph.getGraphString(),
                 strand,
                 seedReasonStr);

  // Get output tensor id before we move the op and invalidate the pointer.
  auto result = modifyRandomSeedOp->output->id(
      ModifyRandomSeedOp::getModifiedSeedOutIndex());

  graph.moveIntoGraph(std::move(modifyRandomSeedOp));

  return result;
}

void RandomSetup::connectOp(Graph &graph,
                            const Config &cfg,
                            const StrandToTensorId &opSeeds,
                            Op *op) const {

  // It would be nice to design this code in a way that did not involve
  // dynamic casts. We deal with RandomBaseOps, CallOps, etc., separately.

  if (auto randomOp = dynamic_cast<RandomBaseOp *>(op)) {
    connectRandomBaseOp(graph, cfg, opSeeds, randomOp);
  } else if (auto callOp = dynamic_cast<CallOp *>(op)) {
    connectSubgraphOp(graph, cfg, opSeeds, callOp, 0, 0);
  } else if (auto loopOp = dynamic_cast<LoopOp *>(op)) {
    connectSubgraphOp(
        graph, cfg, opSeeds, loopOp, 0, LoopOp::getFirstOutputOutIndex());
  } else {
    throw internal_error("[RandomSetup] Random behaviour that requires "
                         "instrumentation of {} ops is currently not "
                         "supported",
                         op->opid);
  }
}

void RandomSetup::connectRandomBaseOp(Graph &graph,
                                      const Config &cfg,
                                      const StrandToTensorId &opSeeds,
                                      RandomBaseOp *op) const {

  // A RandomBaseOp basically just needs it's seed tensor input connecting. We
  // do that here. We also check that the opSeeds map contains a seed only
  // for the strand that the random op is in.

  const auto &opStrand = getStrand(op);
  auto opSeedsIt       = opSeeds.find(opStrand);

  if (opSeeds.size() != 1 || opSeedsIt == opSeeds.end()) {
    // Get strands into a vector so it's easier to add to error message.
    std::vector<Strand> strands;
    for (const auto &entry : opSeeds) {
      strands.push_back(entry.first);
    }

    throw internal_error("[RandomSetup] Expected only a seed for {} for op "
                         "{} (got seeds for {})",
                         opStrand,
                         op->str(),
                         logging::join(strands.begin(), strands.end(), ", "));
  }

  logging::trace("[RandomSetup] Setting seed tensor to {} for {}.",
                 opSeedsIt->second,
                 op->str());

  op->connectInTensor(op->getSeedInIndex(), opSeedsIt->second);
  op->setup();
}

void RandomSetup::connectSubgraphOp(Graph &graph,
                                    const Config &cfg,
                                    const StrandToTensorId &opSeeds,
                                    SubgraphOp *op,
                                    int inputOffset,
                                    int outputOffset) const {

  // LoopOps are responsible for passing through seeds to the loop body so that
  // seeds are available for ops/strands. We have previously added such seeds
  // as inputs and now we need to connect the LoopOps with tensors in opSeeds
  // in accordance to those subgraph inputs. We want the seed to be an explicit
  // inputs that are updated in every loop iteration.
  auto &calledGraph     = op->getCalledGraph();
  const auto &opStrands = cfg.randomOpsMap.at(graph.id).at(op);

  // We have added inputs/outputs to subgraph. This may have messed up inputs
  // and outputs that were already connected to the subgraph op. We need to
  // move affected inputs/outputs to later indices before we connect up the
  // input/output base seeds.
  int moveAmount = opStrands.size();

  // Move inputs.
  int maxIn         = op->input->maxIndex();
  int firstInToMove = cfg.firstSeedInIndexMap.at(calledGraph.id) - inputOffset;
  for (InIndex i = maxIn; i >= firstInToMove; --i) {
    if (op->input->hasIndex(i)) {
      Tensor *t = op->input->tensorMap().at(i);
      op->disconnectInTensor(t);
      op->connectInTensor(i + moveAmount, t->id);
    }
  }

  // Move outputs.
  int maxOut = op->output->maxIndex();
  int firstOutToMove =
      cfg.firstSeedOutIndexMap.at(calledGraph.id) - outputOffset;
  for (InIndex i = maxOut; i >= firstOutToMove; --i) {
    if (op->output->hasIndex(i)) {
      Tensor *t = op->output->tensorMap().at(i);
      op->disconnectOutTensor(t);
      op->connectOutTensor(i + moveAmount, t->id);
    }
  }

  for (const auto &strand : cfg.randomOpsMap.at(graph.id).at(op)) {

    // Get the seed allocated to this strand/op.
    auto opSeedsIt = opSeeds.find(strand);

    if (opSeedsIt == opSeeds.end()) {
      throw internal_error("[RandomSetup] Expected seed for strand {} to be "
                           "available (needed to transform op {}). ",
                           strand,
                           op->str());
    }

    // Work out the id of the graph input.
    auto inBaseSeedId = cfg.inBaseSeedMap.at(calledGraph.id).at(strand);

    if (!calledGraph.hasInputId(inBaseSeedId)) {
      throw internal_error("[RandomSetup] Expected {} to have graph "
                           "input {} (needed to transform op {}). ",
                           calledGraph.getGraphString(),
                           inBaseSeedId,
                           op->str());
    }

    // Connect the tensor to the correct CallOp input.
    InIndex index = calledGraph.getInputIndex(inBaseSeedId) - inputOffset;
    op->connectInTensor(index, opSeedsIt->second);

    // We need to set an output here to make a loop carried dependency. We
    // invent a new tensor name for this, using the op->id value
    // to ensure it's uniqueness. We don't actually use this tensor anywhere.

    auto outBaseSeedId = cfg.outBaseSeedMap.at(calledGraph.id).at(strand);

    if (!calledGraph.hasOutputId(outBaseSeedId)) {
      throw internal_error("[RandomSetup] Expected {} to have graph "
                           "output {} (needed to transform op {}). ",
                           calledGraph.getGraphString(),
                           outBaseSeedId,
                           op->str());
    }

    OutIndex outIndex =
        calledGraph.getOutputIndex(outBaseSeedId) - outputOffset;

    TensorId outId = TensorId(reservedRandomSeedPrefix()) + "_loopcarry" +
                     std::to_string(op->id);
    outId = getTensorIdForStrand(outId, strand);
    outId = graph.addScope(outId);
    op->createAndConnectOutTensor(outIndex, outId);

    logging::trace("[RandomSetup] Passing seed tensor {} to {} in {} "
                   "via {}.",
                   opSeedsIt->second,
                   inBaseSeedId,
                   calledGraph.getGraphString(),
                   op->str());
  }

  op->setup();
}

RandomSetup::OpToStrandToTensorId
RandomSetup::addModifyRandomSeedOps(Graph &graph, const Config &cfg) const {

  OpToStrandToTensorId opSeeds;

  uint32_t modifier = 0u;

  // Add a ModifyRandomSeedOp for every op that needs a random seed in the
  // graph. Note that some ops may need a seed for more than one strand.

  for (const auto &opsToStrandsEntry : cfg.randomOpsMap.at(graph.id)) {
    auto op             = opsToStrandsEntry.first;
    const auto &strands = opsToStrandsEntry.second;

    for (const auto &strand : strands) {
      // Adding op seed for random op/strand.
      auto str = op->str();
      auto out = addModifyRandomSeedOp(
          graph, cfg, strand, modifier++, nonstd::optional<TensorId>(), str);
      // Remember this.
      opSeeds[op][strand] = out;
    }
  }

  // If this isn't the main graph, add a addModifyRandomSeedOp for each strand
  // to populate the output base seed. This can be used by loops as loop
  // carried dependencies.

  if (graph.id != graph.getIr().getMainGraph().id) {

    for (const auto &entry : cfg.outBaseSeedMap.at(graph.id)) {
      const auto &strand      = entry.first;
      const auto &outBaseSeed = entry.second;
      addModifyRandomSeedOp(graph,
                            cfg,
                            strand,
                            modifier++,
                            outBaseSeed,
                            "subgraph's base seed output (may be unused)");
    }
  }

  return opSeeds;
}

bool RandomSetup::hasRandomOps(const Ir &ir) {
  auto ops = ir.getAllOps();
  return std::any_of(ops.begin(), ops.end(), [](const Op *op) {
    return op->isConvertibleTo<RandomBaseOp>();
  });
}

bool RandomSetup::requiresRandomSeed(const Ir &ir) {
  return (RandomSetup::hasRandomOps(ir) ||
          ir.getSessionOptions().enableStochasticRounding);
}

bool RandomSetup::hasRandomSeed(const Ir &ir) {
  return ir.containsTensor(GetRandomSeedOp::getStreamedSeedTensorId());
}

TensorId RandomSetup::getStreamedSeedTensorId() {
  return GetRandomSeedOp::getStreamedSeedTensorId();
}

RandomSetup::Strand RandomSetup::getStrand(const Op *op) {
  Strand key(unusedVGraphId, unusedPipelineStage);
  if (op->hasVirtualGraphId()) {
    std::get<0>(key) = op->getVirtualGraphId();
  }
  if (op->hasPipelineStage()) {
    std::get<1>(key) = op->getPipelineStage();
  }
  return key;
}

TensorId RandomSetup::getTensorIdForStrand(const TensorId &id,
                                           const Strand &strand) {
  auto &virtualGraphId = std::get<0>(strand);
  auto &pipelineStage  = std::get<1>(strand);
  TensorId res         = id;
  if (virtualGraphId >= 0) {
    res += "_vgid" + std::to_string(virtualGraphId);
  }
  if (pipelineStage >= 0) {
    res += "_stage" + std::to_string(pipelineStage);
  }
  return res;
}

std::ostream &operator<<(std::ostream &out, const RandomSetup::Strand &strand) {

  auto &virtualGraphId = std::get<0>(strand);
  auto &pipelineStage  = std::get<1>(strand);
  out << "(vgid=";
  if (virtualGraphId >= 0) {
    out << virtualGraphId;
  } else {
    out << "N/A";
  }
  out << ", stage=";
  if (pipelineStage >= 0) {
    out << pipelineStage;
  } else {
    out << "N/A";
  }
  out << ")";
  return out;
}

namespace {
// RandomSetup.
bool init = Transform::registerTransform(new RandomSetup());
} // namespace

} // namespace popart
