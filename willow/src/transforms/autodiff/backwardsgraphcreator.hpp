// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BACKWARDS_GRAPH_CREATOR_HPP
#define GUARD_NEURALNET_BACKWARDS_GRAPH_CREATOR_HPP

#include <transforms/autodiff/gradgrower.hpp>
#include <popart/bwdgraphinfo.hpp>

namespace popart {

// Forward declaration.
class Op;
class Graph;
class GraphId;
class AutodiffIrInterface;

/**
 * Class that can create backwards graphs.
 **/
class BackwardsGraphCreator : public GradGrower {
public:
  /**
   * Constructor.
   **/
  explicit BackwardsGraphCreator(AutodiffIrInterface &dep);

  /**
   * For a given fwdGraph and an ID for a bwdGraph, create the bwdGraph and
   * return the information required to use it.
   * \param fwdGraph A reference to the forward graph.
   * \param bwdGraphId A proposed identifier for the backwards graph.
   * \param calledGraphsGradInfo Grad information about graphs that are called
   *     by ops in fwdGraph.
   * \return Information about the backward graph.
   **/
  virtual BwdGraphInfo
  createBackwardsGraph(const Graph &fwdGraph,
                       const GraphId &bwdGraphId,
                       const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);
};

} // namespace popart

#endif