// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_AUTODIFF_HPP
#define GUARD_NEURALNET_AUTODIFF_HPP

#include <map>
#include <memory>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/names.hpp>
#include <popart/sessionoptions.hpp>
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
        AutodiffStitchStrategy stitchStrategy);

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
   * Stitch a forward/backward graph pair. That is, make it so that the
   * backwards graph no longer has any non-gradient inputs of forward graph
   * tensors that are neither inputs nor outputs of the forward graph.
   *
   * When autodiff'ing a graph it is PopART's assumption that all input tensors
   * to gradient ops are either 1) an forward op input 2) a forward op output or
   * 3) the gradient of a forward op output. For this to be true for gradients
   * ops associated with ops that have called subgraphs (example: CallOp, IfOp)
   * typically the backwards versions of those called subgraphs must not have
   * inputs that are associated with non-gradient forward tensors that are
   * neither inputs nor outputs of the forward graph. This is because inputs
   * and outputs of forwards (resp. backwards) subgraphs typically map to inputs
   * and outputs of the associated forward op (resp. grad op).
   *
   * For stitch strategies that affect the forward graph's inputs or outputs,
   * this function should also amend all call sites of the forward graph as
   * appropriate. Conversely, for backwards graphs, it is assumed there are no
   * call sites as yet as it's anticipated this method is called before parents
   * of the backward graph exist.
   *
   * NOTE: This function may modify the fwdGraph, bwdGraph or any graphs that
   *    call these graphs, depending on the method. It also may raise an
   *    exception if it is unable to stitch an index.
   *
   * \param ir The IR in the context of which this transformation is applied.
   * \param fwdGraphId The ID of the subgraph to differentiate.
   * \param bwdGraphInfo The data structure describing the bwdGraph.
   * \param stitchStrategy Method by which to stitch any autodiff result for
   *     graphs that are directly or indirectly called by the graph.
   * \param stitchIndices If provided, backwards graph input indices not in this
   *      list must be ignored and backwards graph input indices in this list
   *      must be stitched (or an exception raised). If not set, it's up to the
   *      stitcher what indices to stitch.
   * \return An updated BwdGraphInfo data structure (with some `expectedInputs`
   *     removed).
   **/
  virtual BwdGraphInfo
  stitch(Ir &ir,
         const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         AutodiffStitchStrategy stitchStrategy,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices);

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "Autodiff"; }

private:
  // Helper class to create stitchers (add a setter if it helps with testing).
  std::unique_ptr<StitcherFactory> stitcherFactory;
};

} // namespace popart

#endif
