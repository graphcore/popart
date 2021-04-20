

// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrowergraph.hpp>

#include <transforms/autodiff/backwardsgraphcreator.hpp>
#include <transforms/autodiff/stitcherfactory.hpp>

namespace popart {

GradGrowerGraph::GradGrowerGraph(AutodiffIrInterface &dep)
    : GradGrowerGraphInterface{}, AutodiffHelper{dep},
      stitcherFactory{std::make_unique<StitcherFactory>()} {}

FwdGraphToBwdGraphInfo GradGrowerGraph::growBackwardsGraph(
    const GraphId &fwdGraphId,
    const TensorIds &gradsProvidedForTensors,
    const nonstd::optional<TensorIds> &gradsRequiredForTensors,
    const FwdGraphToBwdGraphInfo &calledGraphResults_,
    Autodiff::StitchStrategy stitchStrategy) {

  // We need to grow called graphs before we grow the main graph.
  BackwardsGraphCreator bwdGraphCreator{dep};
  FwdGraphToBwdGraphInfo calledGraphGradInfo = calledGraphResults_;

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

      // Create a fresh ID.
      GraphId bwdGraphId = bwdGraphCreator.genNewBwdGraphId(fwdGraph.id);

      // Create the backwards graph.
      logging::trace("[Autodiff] Creating backwards graph '{}'", bwdGraphId);
      auto bwdGraphGradInfo = bwdGraphCreator.createBackwardsGraph(
          fwdGraph, bwdGraphId, calledGraphGradInfo);

      // Stitch backwards graph inputs.
      logging::trace("[Autodiff] Stitching backwards graph '{}'", bwdGraphId);
      // Don't set filter, stitch all inputs.
      bwdGraphGradInfo = stitcher->stitch(fwdGraph.id, bwdGraphGradInfo, {});

      // Store the result info for parents graphs.
      calledGraphGradInfo.insert({fwdGraph.id, bwdGraphGradInfo});
    }
  }

  return calledGraphGradInfo;
}

} // namespace popart
