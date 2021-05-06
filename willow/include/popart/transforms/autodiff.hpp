// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_AUTODIFF_HPP
#define GUARD_NEURALNET_AUTODIFF_HPP

#include <map>
#include <memory>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/names.hpp>
#include <popart/transforms/transform.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

// Forward declarations.
class AutodiffIrInterface;
class StitcherFactory;

/**
 * Class responsible for auto-differentiation.
 **/
class Autodiff : public Transform {
public:
  /**
   * Type representing a method by which ensure that a bwdGraph's inputs are
   * exclusively fwdGraph inputs, fwdGraph outputs or gradients of fwdGraph
   * outputs.
   **/
  enum class StitchStrategy {
    /// Recompute all fwd tensors except those that are inputs in the fwdGraph.
    Recompute = 0,
    /// Number of StitchStrategy values.
    N = 1
  };

  static std::size_t id();

  Autodiff();
  virtual ~Autodiff() override;

  // Shorthand.
  using TensorIds  = std::vector<TensorId>;
  using FwdGraphId = GraphId;

  /**
   * Implementation of Transform::apply. Will apply autodiff to whole IR.
   */
  virtual bool apply(Graph &graph) const final;

  /**
   * Apply autodiff to entire IR.
   */
  virtual bool applyToIr(Ir &ir) const final;

  /**
   * Apply `createBwdGraph` and `stitch` recursively, top-down, resulting in
   * the creation of a backwards graph for fwdGraphId. This may recursively
   * create the backwards graphs for any graphs that aren't already autodiff'ed
   * but for which a backwards graph is required because they are called.
   *
   * NOTE: This method may fail if some required gradient cannot be produced.
   *
   * \param ir The IR in the context of which this transformation is applied.
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
  apply(Ir &ir,
        const GraphId &fwdGraphId,
        const TensorIds &gradsProvidedForFwdId,
        const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
        const FwdGraphToBwdGraphInfo &calledGraphsGradInfo,
        StitchStrategy stitchStrategy);

  /**
   * Create a backwards graph plus info required for creation of backwards
   * graphs for parent graphs for a specific subgraph only (non-recursive).
   * This method returns an "unstitched" result. That is, it is not guaranteed
   * that all non-gradient inputs to a backwards graph are available as inputs
   * or outputs of the forward graph. This is a precondition for BwdGraphInfo
   * objects used as values in `calledGraphsGradInfo` so you must call `stitch`
   * to stitch the result before using the result info in another autodiff call.
   *
   * NOTE: This method may fail if some required gradient cannot be produced.
   *
   * \param ir The IR in the context of which this transformation is applied.
   * \param fwdGraphId The ID of the subgraph to differentiate.
   * \param gradsProvidedForFwdId The tensors (normally outputs of
   *     fwdGraph) for which gradient tensors are available.
   * \param gradsRequiredForFwdId The tensors (normally inputs of the
   *     fwdGraph) for which gradient tensors are required (as outputs to the
   *     returned backwards graph).
   * \param calledGraphsGradInfo The result information from applying autodiff
   *     for the graphs that are called by subgraph ops in fwdGraph. It is a
   *     precondition of this function that the graphs provided in this map
   *     are stitched.
   * \return An BwdGraphInfo object with the following properties:
   *     - `expectedInputs` may contain arbitrary tuples
   *       `(t, ExpectedConnectionType::Fwd)` where `t` is any tensor in the
   *       forward graph (it need not be an input or output). Only tensors `t`
   *       in `gradsProvidedForFwdId` may appear as a tuple
   *       `(t, ExpectedConnectionType::FwdGrad)` in `expectedInputs`.
   *     - `expectedOutputs` may only contain tuples of the type
   *       `(t, ExpectedConnectionType::FwdGrad)` where `t` is an input tensor
   *       of the forward graph. If set, each tensor `t` in
   *       `gradsRequiredForFwdId` must be present in `expectedOutputs`.
   **/
  virtual BwdGraphInfo
  createBwdGraph(Ir &ir,
                 const GraphId &fwdGraphId,
                 const TensorIds &gradsProvidedForFwdId,
                 const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
                 const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);

  /**
   * Stitch a fwd/bwd graph pair. That is, remove some non-gradient inputs from
   * the backwards graph if possible. This function can be used to make it so
   * so that all inputs to the backwards graph are either inputs of the forward
   * graph, outputs of the forward graph or gradients of outputs of the forward
   * graph (this is a requirement for growing some grad ops) using a method of
   * the caller's choosing.
   *
   * NOTE: This function may modify the fwdGraph, bwdGraph or any graphs that
   *    call these graphs, depending on the method.
   *
   * \param ir The IR in the context of which this transformation is applied.
   * \param fwdGraphId The ID of the subgraph to differentiate.
   * \param bwdGraphInfo The data structure describing the bwdGraph.
   * \param stitchStrategy Method by which to stitch any autodiff result for
   *     graphs that are directly or indirectly called by the graph.
   * \param stitchIndices If provided, stitching is constrained to backward
   *     graph input indices in this list. If not provided, all non-gradient
   *     inputs of the backwards graph are considered for stitching.
   * \return An updated BwdGraphInfo data structure (with some `expectedInputs`
   *     removed).
   **/
  virtual BwdGraphInfo
  stitch(Ir &ir,
         const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         StitchStrategy stitchStrategy,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices);

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Autodiff"; }

private:
  // Helper class to create stitchers (add a setter if it helps with testing).
  std::unique_ptr<StitcherFactory> stitcherFactory;
};

} // namespace popart

#endif
