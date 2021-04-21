// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BACKWARDS_GRAPH_CREATOR_HELPER_HPP
#define GUARD_NEURALNET_BACKWARDS_GRAPH_CREATOR_HELPER_HPP

#include <popart/bwdgraphinfo.hpp>
#include <popart/names.hpp>

#include "popart/patterns/patterns.hpp"
#include <transforms/autodiff/tensorgradmapregister.hpp>

#include <map>
#include <memory>
#include <vector>

namespace popart {

// Forward declaration.
class Op;
class Graph;

/**
 * Class to help populate a backwards graphs. This helper is supposed to be
 * instantiated once for every fwd graph. Note that the actual work is done
 * in the call to `populateBwdGraph`, not in the constructor.
 **/
class BackwardsGraphCreatorHelper {
public:
  /**
   * Constructor.
   **/
  BackwardsGraphCreatorHelper(
      const Graph &fwdGraph,
      Graph &bwdGraph,
      const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);

  /**
   * Function that populates the bwdGraph passed in the constructor.
   * \return Return information about the backwards graph.
   */
  virtual BwdGraphInfo populateBwdGraph();

private:
  void growGradGraph();
  std::vector<Op *> growGradOps(Op *nonGradOp);
  bool opIsReadyToCreateGradients(Op *);
  std::string opNotReadyExplanation(Op *op);
  void registerBwdOp(Op *fwdOp, Op *bwdOp);
  Op *growGradSumOp(Tensor *target, const std::vector<Tensor *> &partials);
  // Check if tensor in bwdGraph is a gradient tensor.
  bool bwdIdIsGrad(const TensorId &);
  // Check if tensor in bwdGraph is a non-gradient tensor.
  bool bwdIdIsNonGrad(const TensorId &);
  // Convert fwdGraph tensor to a gradient tensor in bwdGraph.
  TensorId fwdIdToBwdGradId(const TensorId &);
  // Convert a bwdGraph gradient tensor into a fwdGraph gradient tensor.
  TensorId bwdGradIdToFwdId(const TensorId &);
  // Convert a bwdGraph gradient tensor into a fwdGraph non-gradient tensor.
  TensorId bwdNonGradIdToFwdId(const TensorId &);
  BwdGraphInfo makeGradInfo();
  bool hasInputTensorId(Op *nonGradOp, const GradInOutMapper &inOutMapper);
  TensorId getInputTensorId(Op *nonGradOp, const GradInOutMapper &inOutMapper);

  static void cloneGraph(const Graph &from, Graph &to);
  static void doPrune(Graph &);

  const Graph &fwdGraph;
  Graph &bwdGraph;
  const FwdGraphToBwdGraphInfo &calledGraphsGradInfo;

  // A map of fwd tensors to their corresponding gradient tensors
  std::map<TensorId, TensorId> gradTensorMap;
  TensorGradMapRegister gradRegister;
  std::map<Op *, std::vector<std::unique_ptr<Op>>> gradOpStore;
};

} // namespace popart

#endif