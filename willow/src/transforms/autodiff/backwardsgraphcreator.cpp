// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/backwardsgraphcreator.hpp>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>

#include <transforms/autodiff/autodiffirinterface.hpp>
#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>

namespace popart {

BackwardsGraphCreator::BackwardsGraphCreator(AutodiffIrInterface &dep)
    : GradGrower(dep) {}

BwdGraphInfo BackwardsGraphCreator::createBackwardsGraph(
    const Graph &fwdGraph,
    const GraphId &bwdGraphId,
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {

  if (dep.get().hasGraph(bwdGraphId)) {
    throw error("[Autodiff] Unable to create new backwards graph for {} with "
                "GraphId '{}' because a graph with this ID already exists",
                fwdGraph.getGraphString(),
                bwdGraphId);
  }

  Graph &bwdGraph = dep.get().createGraph(bwdGraphId);
  BackwardsGraphCreatorHelper helper(fwdGraph, bwdGraph, calledGraphsGradInfo);
  return helper.populateBwdGraph();
}

} // namespace popart