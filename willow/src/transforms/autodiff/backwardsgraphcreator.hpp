// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BACKWARDS_GRAPH_CREATOR_HPP
#define GUARD_NEURALNET_BACKWARDS_GRAPH_CREATOR_HPP

#include <transforms/autodiff/autodiffhelper.hpp>
#include <vector>
#include <popart/bwdgraphinfo.hpp>
#include <popart/vendored/optional.hpp> // IWYU pragma: keep

#include "popart/names.hpp"

namespace popart {

// Forward declaration.

class Graph;
class GraphId;
class AutodiffIrInterface;

/**
 * Class that can create backwards graphs.
 **/
class BackwardsGraphCreator : public AutodiffHelper {
public:
  // Shorthand.
  using TensorIds = std::vector<TensorId>;

  /**
   * Constructor.
   **/
  explicit BackwardsGraphCreator(AutodiffIrInterface &dep);

  /**
   * For a given fwdGraph and an ID for a bwdGraph, create the bwdGraph and
   * return the information required to use it.
   * \param fwdGraph A reference to the forward graph.
   * \param bwdGraphId A proposed identifier for the backwards graph.
   * \param gradsProvidedForFwdId The tensors (normally outputs of
   *     fwdGraph) for which gradient tensors are available.
   * \param gradsRequiredForFwdId  The tensors (normally inputs of the
   *     fwdGraph) for which gradient tensors are required (as outputs to the
   *     returned backwards graph).
   * \param calledGraphsGradInfo Grad information about graphs that are called
   *     by ops in fwdGraph.
   * \return Information about the backward graph.
   **/
  virtual BwdGraphInfo
  createBackwardsGraph(const Graph &fwdGraph,
                       const GraphId &bwdGraphId,
                       const nonstd::optional<TensorIds> &gradsProvidedForFwdId,
                       const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
                       const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);

  /**
   * Create a new, unused ID for the backwards graph.
   **/
  GraphId genNewBwdGraphId(const GraphId &fwdGraphId) const;
};

} // namespace popart

#endif
