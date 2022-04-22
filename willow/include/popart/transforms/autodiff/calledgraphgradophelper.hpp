// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CALLED_GRAPH_GRAD_OP_HELPER_HPP
#define GUARD_NEURALNET_CALLED_GRAPH_GRAD_OP_HELPER_HPP

#include <functional>
#include <map>
#include <vector>
#include <popart/bwdgraphinfo.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"

namespace popart {
class GradInOutMapper;
class Graph;
class Op;

/**
 * Helper class that helps ops that have called subgraphs with handling for
 * backwards versions of said called subgraphs.
 *
 *  - Remembers the parameter of `setCalledSubgraphGradInfo` calls in a member
 *    called `calledGraphsGradInfo`.
 *  - Provides helper functions to generate the information required to connect
 *    inputs and outputs of grad ops (with some documented assumptions).
 **/
class CalledGraphGradOpHelper {
public:
  // Constructor.
  CalledGraphGradOpHelper(Op *op);
  // Destructor.
  virtual ~CalledGraphGradOpHelper();

  /**
   * Method for ops to call on their Op::setCalledSubgraphGradInfo call.
   * \param calledGraphsGradInfo An object describing information related to
   *     called graphs for the op.
   **/
  virtual void
  setCalledSubgraphGradInfo(const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);

  /**
   * Retrieve parameter of last call to `setCalledSubgraphGradInfo` (or throws
   * an exception if no such call happened).
   * \returns The argument of the last call to `setCalledSubgraphGradInfo`.
   */
  virtual const FwdGraphToBwdGraphInfo &getCalledSubgraphGradInfo() const;

  /**
   * Function that returns the bwd graph for a called graph (or throws an
   * exception if it is unable to find it).
   *
   * NOTE: It is assumed this function is used by getGradOps() and hence
   * `setCalledSubgraphGradInfo` has been called. If this is not the case
   * an exception will be thrown.
   **/
  virtual Graph &getBwdGraph(SubgraphIndex subgraphIndex);

  /**
   * Helper function to get the BwdGraphInfo for a called graph.
   *
   * NOTE: It is assumed this function is used by getGradOps() and hence
   * `setCalledSubgraphGradInfo` has been called. If this is not the case
   * an exception will be thrown.
   **/
  virtual const BwdGraphInfo &getBwdGraphInfo(SubgraphIndex subgraphIndex);

  /**
   * Helper function that determines for every required input of the bwd graph
   * of a called graph a structure that describes which a) input of this op
   * or b) output of this op or c) gradient of an output should be connected
   * to this input.
   *
   * NOTE: It is assumed this function is used by getGradOps() and hence
   * `setCalledSubgraphGradInfo` has been called. If this is not the case
   * an exception will be thrown.
   *
   * NOTE: The output of this function is based on inputs of the bwdGraph --
   * if a grad op requires additional inputs that are not based on bwdGraph
   * inputs then those will not be included here (example: condition for IfOp).
   **/
  virtual std::vector<GradInOutMapper> getCalledGraphGradInInfo(
      SubgraphIndex subgraphIndex,
      const std::function<InIndex(InIndex)> &bwdGraphInToGradOpInIndex);

  /**
   * Helper function that determines for every provided output of the bwd graph
   * of a called graph for this op, the which of this op's inputs is associated
   * with the gradient said output produces.
   *
   * NOTE: It is assumed this function is used by getGradOps() and hence
   * `setCalledSubgraphGradInfo` has been called. If this is not the case
   * an exception will be thrown.
   *
   * NOTE: The output of this function is based on outputs of the bwdGraph --
   * if a grad op prodices additional outputs that are not based on bwdGraph
   * outputs then those will not be included here.
   **/
  virtual const std::map<int, int> getCalledGraphGradOutToNonGradIn(
      SubgraphIndex subgraphIndex,
      const std::function<OutIndex(OutIndex)> &bwdGraphOutToGradOpOutIndex);

protected:
  // Helper function.
  void throwExceptionIfInfoUnavailable() const;

  // Op that's using this.
  Op *op;
  // Latest received data via setCalledSubgraphGradInfo.
  nonstd::optional<FwdGraphToBwdGraphInfo> calledGraphsGradInfo;
};

} // namespace popart

#endif
