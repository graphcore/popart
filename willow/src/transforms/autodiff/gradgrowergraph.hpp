// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAD_GROWER_GRAPH_HPP
#define GUARD_NEURALNET_GRAD_GROWER_GRAPH_HPP

#include <functional>
#include <memory>

#include <popart/names.hpp>

#include <popart/transforms/autodiff.hpp>

#include <transforms/autodiff/autodiffhelper.hpp>

namespace popart {

// Forward declaration.
class StitcherFactory;

/**
 * Interface for GradGrowerSubgraph.
 */
class GradGrowerGraphInterface {
public:
  // Shorthand.
  using TensorIds = std::vector<TensorId>;

  // Grow backwards pass for subgraph and all descendants.
  virtual FwdGraphToBwdGraphInfo
  growBackwardsGraph(const GraphId &fwdGraphId,
                     const nonstd::optional<TensorIds> &gradsProvidedForTensors,
                     const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
                     const FwdGraphToBwdGraphInfo &calledGraphsGradInfo,
                     AutodiffStitchStrategy stitchStrategy) = 0;

  virtual ~GradGrowerGraphInterface() {}
};

/**
 * Helper class for recursively growing backwards subgraphs.
 */
class GradGrowerGraph : public GradGrowerGraphInterface,
                        private AutodiffHelper {
public:
  // Constructor.
  GradGrowerGraph(AutodiffIrInterface &dep);
  ~GradGrowerGraph();

  /**
   * Apply autodiff recursively, top-down, resulting in the creation of a
   * backwards graph for fwdGraphId. This may recursively create the backwards
   * graphs for any graphs that aren't already autodiff'ed but for which a
   * backwards graph is required because they are called.
   *
   * NOTE: This method may fail if some required gradient cannot be produced.
   *
   * \param fwdGraphId The ID of the subgraph to differentiate.
   * \param gradsProvidedForFwdId The tensors (normally outputs of
   *     fwdGraph) for which gradient tensors are available.
   * \param gradsRequiredForFwdId If set, the tensors the users requires
   *     gradients for (normally inputs of fwdGraph). If autodiff is unable to
   *     provide the required gradient an error will be raised. If unset,
   *     autodiff will provide as many gradients as possible.
   * \param calledGraphsGradInfo The result information from applying autodiff
   *     for the graphs that are called by subgraph ops in fwdGraph. It is a
   *     precondition of this function that the graphs provided in this map
   *     are stitched.
   * \param stitchStrategy Method by which to stitch any autodiff result for
   *     graphs that are directly or indirectly called by the graph. This stitch
   *     strategy will be universally applied to all relevant inputs.
   * \return An FwdGraphToBwdGraphInfo object that contains BwdGraphInfo for
   *     all descended Graphs and for which all entries have the following
   *     properties:
   *     - `expectedInputs` may contain a tuple
   *       `(t, ExpectedConnectionType::Fwd)` iff `t` is an input or output
   *       tensor of the forward graph. Only tensors `t` in
   *       `gradsProvidedForFwdId` may appear as a tuple
   *       `(t, ExpectedConnectionType::FwdGrad)` in `expectedInputs`.
   *     - `expectedOutputs` may only contain tuples of the type
   *       `(t, ExpectedConnectionType::FwdGrad)` where `t` is an input tensor
   *       of the forward graph. If set, each tensor `t` in
   *       `gradsRequiredForFwdId` must be present in `expectedOutputs`.
   */
  virtual FwdGraphToBwdGraphInfo
  growBackwardsGraph(const GraphId &fwdGraphId,
                     const nonstd::optional<TensorIds> &gradsProvidedForTensors,
                     const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
                     const FwdGraphToBwdGraphInfo &calledGraphsGradInfo,
                     AutodiffStitchStrategy stitchStrategy) override;

private:
  // Helper class to create stitchers (add a setter if it helps with testing).
  std::unique_ptr<StitcherFactory> stitcherFactory;
};

} // namespace popart

#endif
