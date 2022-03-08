// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrowergraph.hpp>

#include <popart/bwdgraphinfo.hpp>

#include <transforms/autodiff/backwardsgraphcreator.hpp>
#include <transforms/autodiff/stitcherfactory.hpp>

namespace popart {
namespace {
/**
 * Check the graph for modifying ops. Autodiff does not work on modifying ops,
 * e.g. those created explicitly by the user in popXL.
 *
 * \param graph The graph to check.
 */
void checkForModifyingOps(Graph &graph) {
  for (auto &opPair : graph.getOps()) {
    if (opPair.second->modifies()) {
      throw error("The graph {} contains a modifying op: {}. Autodiff cannot "
                  "be used on graphs with inplace modifying ops.",
                  graph.id,
                  opPair.second->debugName());
    }
  }
}
} // namespace

GradGrowerGraph::GradGrowerGraph(AutodiffIrInterface &dep)
    : GradGrowerGraphInterface{}, AutodiffHelper{dep},
      stitcherFactory{std::make_unique<StitcherFactory>()} {}

FwdGraphToBwdGraphInfo GradGrowerGraph::growBackwardsGraph(
    const GraphId &fwdGraphId,
    const nonstd::optional<TensorIds> &gradsProvidedForTensors,
    const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo_,
    AutodiffStitchStrategy stitchStrategy) {

  // We need to grow called graphs before we grow the main graph.
  BackwardsGraphCreator bwdGraphCreator{dep};
  FwdGraphToBwdGraphInfo calledGraphGradInfo = calledGraphsGradInfo_;

  // Get stitcher.
  auto stitcher = stitcherFactory->createStitcher(dep, stitchStrategy);

  // Bottom-up schedule limited to our call hierarchy.
  auto graphSched = dep.get().getGraphSchedule(fwdGraphId);

  // Apply autodiff in reverse, top-down (i.e. children before parents).
  for (auto fwdGraphIt = graphSched.rbegin(); fwdGraphIt != graphSched.rend();
       ++fwdGraphIt) {

    // Get a non-const reference to the fwdGraph.
    auto &fwdGraph = dep.get().getGraph((*fwdGraphIt)->id);

    // Check if we don't already have a backwards graph for this subgraph.
    if (calledGraphGradInfo.find(fwdGraph.id) == calledGraphGradInfo.end()) {

      // Check this graph has no modifying ops that would cause problems in
      // autodiff.
      checkForModifyingOps(fwdGraph);

      // Create a fresh ID.
      GraphId bwdGraphId = bwdGraphCreator.genNewBwdGraphId(fwdGraph.id);

      // Create the backwards graph.
      logging::trace("[Autodiff] Creating backwards graph '{}'", bwdGraphId);

      // For all but top-level graph, use whichever gradients of forward graph
      // outputs we need to produce whatever gradients of forward graph inputs
      // we can. We do this by passing in nullopt.
      auto providedGrads = (fwdGraph.id == fwdGraphId) ? gradsProvidedForTensors
                                                       : nonstd::nullopt;

      // Only specify required tensors for the top-level graph (the user only
      // provides this information for the top-level graph, and we dont know for
      // subgraphs so we default to nullopt).
      auto requiredGrads =
          (fwdGraph.id == fwdGraphId) ? gradsRequiredForFwdId : nonstd::nullopt;

      auto bwdGraphGradInfo =
          bwdGraphCreator.createBackwardsGraph(fwdGraph,
                                               bwdGraphId,
                                               providedGrads,
                                               requiredGrads,
                                               calledGraphGradInfo);

      // Stitch backwards graph inputs.
      logging::trace("[Autodiff] Stitching backwards graph '{}'", bwdGraphId);
      // Don't set stitch inputs, rely on default behaviour.
      bwdGraphGradInfo = stitcher->stitch(fwdGraph.id, bwdGraphGradInfo, {});

      // Store the result info for parents graphs.
      calledGraphGradInfo.insert({fwdGraph.id, bwdGraphGradInfo});
    }
  }

  return calledGraphGradInfo;
}

} // namespace popart
