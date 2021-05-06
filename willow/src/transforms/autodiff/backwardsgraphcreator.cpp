// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/backwardsgraphcreator.hpp>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>

#include <transforms/autodiff/autodiffirinterface.hpp>
#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>

namespace popart {

BackwardsGraphCreator::BackwardsGraphCreator(AutodiffIrInterface &dep)
    : AutodiffHelper(dep) {}

BwdGraphInfo BackwardsGraphCreator::createBackwardsGraph(
    const Graph &fwdGraph,
    const GraphId &bwdGraphId,
    const TensorIds &gradsProvidedForFwdId,
    const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {

  if (dep.get().hasGraph(bwdGraphId)) {
    throw error("[Autodiff] Unable to create new backwards graph for {} with "
                "GraphId '{}' because a graph with this ID already exists",
                fwdGraph.getGraphString(),
                bwdGraphId);
  }

  Graph &bwdGraph = dep.get().createGraph(bwdGraphId);
  BackwardsGraphCreatorHelper helper(fwdGraph, bwdGraph);
  return helper.populateBwdGraph(
      gradsProvidedForFwdId, gradsRequiredForFwdId, calledGraphsGradInfo);
}

GraphId
BackwardsGraphCreator::genNewBwdGraphId(const GraphId &fwdGraphId) const {

  GraphId result = GraphId(logging::format("{}_bwd", fwdGraphId));

  static int64_t counter = 0ll;
  while (dep.get().hasGraph(result)) {
    // Graph already contains result id, try a new one. This could be expensive
    // if repeated a lot of times, but generally there is no need to create
    // the autodiff of a graph twice, so this probably won't be used much.
    result = GraphId(logging::format("{}_bwd_{}", fwdGraphId, counter++));
  }

  return result;
}

} // namespace popart