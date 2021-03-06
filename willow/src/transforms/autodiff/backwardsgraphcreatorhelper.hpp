// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_BACKWARDS_GRAPH_CREATOR_HELPER_HPP
#define GUARD_NEURALNET_BACKWARDS_GRAPH_CREATOR_HELPER_HPP

#include <popart/bwdgraphinfo.hpp>
#include <popart/names.hpp>

#include <popart/graph.hpp>
#include <popart/patterns/patterns.hpp>

#include <transforms/autodiff/tensorgradmapregister.hpp>

#include <map>
#include <memory>
#include <vector>

namespace popart {

// Forward declaration.
class Op;
class CopyInputMarkings;
class CopyOutputMarkings;

/**
 * Class to help populate a backwards graphs. This helper is supposed to be
 * instantiated once for every fwd graph. Note that the actual work is done
 * in the call to `populateBwdGraph`, not in the constructor.
 **/
class BackwardsGraphCreatorHelper {
public:
  // Shorthand.
  using TensorIds = std::vector<TensorId>;

  /**
   * Constructor.
   **/
  BackwardsGraphCreatorHelper(const Graph &fwdGraph, Graph &bwdGraph);

  /**
   * Function that populates the bwdGraph passed in the constructor.
   * \param gradsRequiredForFwdId  The tensors (normally inputs of the
   *     fwdGraph) for which gradient tensors are required (as outputs to the
   *     returned backwards graph).
   * \param calledGraphsGradInfo The result information from applying autodiff
   *     for the graphs that are called by subgraph ops in fwdGraph. It is a
   *     precondition of this function that the graphs provided in this map
   *     are stitched.
   * \return Return information about the backwards graph.
   */
  virtual BwdGraphInfo
  populateBwdGraph(const TensorIds &gradsProvidedForFwdId,
                   const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
                   const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);

  virtual BwdGraphInfo makeGradInfo();

  static void doPrune(Graph &);

private:
  void growGradGraph(const TensorIds &gradsProvidedForFwdId,
                     const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
                     const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);
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
  bool hasInputTensorId(Op *nonGradOp, const GradInOutMapper &inOutMapper);
  TensorId getInputTensorId(Op *nonGradOp, const GradInOutMapper &inOutMapper);

  const Graph &fwdGraph;
  Graph &bwdGraph;

  // A map of fwd tensors to their corresponding gradient tensors
  std::map<TensorId, TensorId> gradTensorMap;
  TensorGradMapRegister gradRegister;
  std::map<Op *, std::vector<std::unique_ptr<Op>>> gradOpStore;
};

} // namespace popart

#endif