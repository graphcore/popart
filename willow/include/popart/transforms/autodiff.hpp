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
 * Class responsible for the automatic differentiation (autodiff) transform.
 **/
class Autodiff : public Transform {
public:
  /**
   * ID of the autodiff transform.
   */
  static std::size_t id();

  /**
   * Default constructor for the Autodiff class.
   **/
  Autodiff();

  /**
   * Destructor for the Autodiff class.
   **/
  virtual ~Autodiff() override;

  /**
   * Vector of tensor IDs.
   **/
  using TensorIds = std::vector<TensorId>;

  /**
   * ID of the forward graph.
   **/
  using FwdGraphId = GraphId;

  /**
   * Perform automatic differentiation.
   *
   * Implemented as `applyToIr(graph.getIr()))`
   *
   * \param graph The `autodiff` transform is applied to the IR containing the
   *     Graph `graph`.
   *
   * \return An indication of whether the automatic differentiation has been
   *      completed (`true`) or not (`false`).
   */
  virtual bool apply(Graph &graph) const override;

  /**
   * Perform automatic differentiation.
   *
   * \param ir The IR to apply the `autodiff` transform to.
   *
   * \return An indication of whether the automatic differentiation has been
   *      completed (`true`) or not (`false`).
   */
  virtual bool applyToIr(Ir &ir) const;

  /**
   * Create a backward graph.
   *
   * Apply `createBwdGraph()` and `stitch()` recursively, top-down, to create a
   * backward graph for the forward graph with ID `fwdGraphId`.
   *
   * The forward graph being differentiated can call subgraphs. If the
   * `autodiff` transform has already been applied to the subgraphs and the
   * result stored in \p `calledGraphsGradInfo`, then the backward graph that
   * has already been created for the subgraphs will be used. Otherwise, this
   * method will recurse on the subgraphs.
   *
   * When recursing on a subgraph, this method does not know for which tensors
   * gradients are required. If a null \p gradsRequiredForFwdId is passed, the
   * `autodiff` transform will produce gradients for all input tensors.
   *
   * For control over which gradients are produced for the subgraph, first
   * (manually) call the `autodiff` transform on the subgraph and pass \p
   * gradsRequiredForFwdId. Store the resultant `BwdGraphInfo` in the \p
   * FwdGraphToBwdGraphInfo map passed to the `autodiff` call for the forward
   * graph.
   *
   * NOTE: This method may fail if any required gradient cannot be produced.
   *
   * \param ir The IR to which this transform is applied.
   * \param fwdGraphId The ID of the graph to differentiate.
   * \param gradsProvidedForFwdId An optional list of tensors (normally outputs
   *      of the forward graph) for which gradient tensors are available to be
   *      used as inputs to the backward graph. If set, the `autodiff`
   *      transform will make the gradients of these forward tensors the first
   *      inputs of the the backward graph. If not set, the `autodiff`
   *      transform will use whatever gradients of outputs of the forward graph
   *      it needs as outputs of the backward graph to allow the backward graph
   *      to produce the gradients that are required.
   * \param gradsRequiredForFwdId The IDs of the forward graph tensors for
   *      which the backward graph should produce gradients. If set, the
   *      backward graph will compute only the gradients of the specified
   *      tensors and mark them as outputs. If one of these gradients cannot be
   *      computed, it is an error. If not set, the backward graph will
   *      produce as many gradients of the forward graph inputs as possible,
   *      and mark these all as outputs. If set, but an empty vector is passed,
   *      this is an error, as you are requesting no gradients be computed at
   *      all, resulting in an empty graph.
   * \param calledGraphsGradInfo The result of applying the `autodiff`
   *      transform to the graphs that are called by subgraph ops in the
   *      forward graph. It is a precondition of this method that the graphs
   *      provided in this map are stitched.
   * \param stitchStrategy The method used to stitch any result of the
   *      `autodiff` transform for graphs that are directly or indirectly
   *      called by the graph. This stitch strategy will be universally applied
   *      to all relevant inputs.
   * \return An `FwdGraphToBwdGraphInfo` object that contains `BwdGraphInfo` for
   *      all descended graphs and for which all entries have the following
   *      properties:
   *     - `expectedInputs` may contain a tuple
   *       `(t, ExpectedConnectionType::Fwd)` iff `t` is an input or output
   *       tensor of the forward graph. Only tensors `t` in
   *       \p gradsProvidedForFwdId may appear as a tuple
   *       `(t, ExpectedConnectionType::FwdGrad)` in `expectedInputs`.
   *       If \p gradsProvidedForFwdId is set, the first inputs will match the
   *       gradients of \p gradsProvidedForFwdId, respecting the order.
   *     - `expectedOutputs` may only contain tuples of the type
   *       `(t, ExpectedConnectionType::FwdGrad)` where `t` is an input tensor
   *       of the forward graph. If `gradsRequiredForFwdId` is set, the
   *       `expectedOutputs` list matches the size and order of
   *       `gradsRequiredForFwdId` exactly. If not set, the list is ordered
   *       in the order of the forward graph inputs, although some gradients
   *       of the forward graph inputs may be missing.
   */
  virtual FwdGraphToBwdGraphInfo
  apply(Ir &ir,
        const GraphId &fwdGraphId,
        const nonstd::optional<TensorIds> &gradsProvidedForFwdId,
        const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
        const FwdGraphToBwdGraphInfo &calledGraphsGradInfo,
        AutodiffStitchStrategy stitchStrategy);

  /**
   * Create backward graph information for a specific subgraph
   * (non-recursive).
   *
   * This method returns an "unstitched" result. This means that it is not
   * guaranteed that all non-gradient inputs to a backward graph are available
   * as inputs or outputs of the forward graph. This is a precondition for
   * `BwdGraphInfo` objects used as values in `calledGraphsGradInfo`. So, you
   * must call `stitch` on the result before using the result information in an
   * `autodiff` call.
   *
   * NOTE: This method may fail if any required gradient cannot be produced.
   *
   * \param ir The IR to which this transform is applied.
   * \param fwdGraphId The ID of the subgraph to differentiate.
   * \param gradsProvidedForFwdId An optional list of tensors (normally outputs
   *     of the forward graph) for which gradient tensors are available to be
   *     used as inputs to the backward graph.
   *     If set, autodiff will make the gradients of these forward tensors the
   *     first inputs of the the backward graph.
   *     If not set, autodiff will use whatever gradients of outputs of the
   *     forward graph it needs as outputs of the backward graph to allow the
   *     backward graph to produce the gradients that are required.
   * \param gradsRequiredForFwdId The IDs of the forward graph tensors
   *     for which the backward graph should produce gradients.
   *     If set, the backward graph will compute only the gradients of the
   *     specified tensors and mark them as outputs. If one of these gradients
   *     cannot be computed, it is an error.
   *     If not set, the backward graph will produce as many gradients of the
   *     forward graph inputs as possible, and mark all these as outputs.
   *     If set, but an empty vector is passed, this is an error, as you are
   *     requesting no gradients be computed at all, resulting in an empty
   *     graph.
   * \param calledGraphsGradInfo The result of applying autodiff to the
   *     graphs that are called by subgraph ops in the forward graph. It is a
   *     precondition of this method that the graphs provided in this map
   *     are stitched.
   * \return A `BwdGraphInfo` object with the following properties:
   *     - `expectedInputs` may contain arbitrary tuples
   *       `(t, ExpectedConnectionType::Fwd)` where `t` is any tensor in the
   *       forward graph (it need not be an input or output). Only tensors `t`
   *       in \p gradsProvidedForFwdId may appear as a tuple
   *       `(t, ExpectedConnectionType::FwdGrad)` in `expectedInputs`.
   *       If \p gradsProvidedForFwdId is set, the first inputs will match the
   *       gradients of \p gradsProvidedForFwdId, respecting the order.
   *     - `expectedOutputs` may only contain tuples of the type
   *       `(t, ExpectedConnectionType::FwdGrad)` where `t` is an input tensor
   *       of the forward graph. If `gradsRequiredForFwdId` is set, the
   *       `expectedOutputs` list matches the size and order of
   *       `gradsRequiredForFwdId` exactly. If not set, the list is ordered
   *       in the order of the forward graph inputs, although some gradients
   *       of the forward graph inputs may be missing.
   **/
  virtual BwdGraphInfo
  createBwdGraph(Ir &ir,
                 const GraphId &fwdGraphId,
                 const nonstd::optional<TensorIds> &gradsProvidedForFwdId,
                 const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
                 const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);

  /**
   * Stitch a forward-backward graph pair.
   *
   * To stitch a forward-backward graph pair means to make it so that the
   * backward graph no longer has any non-gradient inputs of the forward graph
   * tensors that are neither inputs nor outputs of the forward graph.
   *
   * When applying the `autodiff` transform to a graph, PopART assumes that all
   * input tensors to the gradient ops are either 1) a forward op input 2) a
   * forward op output or 3) the gradient of a forward op output. For this to be
   * true for gradient ops of subgraph ops (for example: `CallOp` and `IfOp`),
   * typically the backward graphs of those called subgraphs must not have
   * inputs that are associated with non-gradient forward tensors that are
   * neither inputs nor outputs of the forward graph. This is because the inputs
   * and outputs of a forward subgraph typically map to the inputs and outputs
   * of the associated forward op. Similarly, the inputs and outputs of a
   * backward subgraph typically map to the inputs and outputs of the associated
   * gradient op.
   *
   * For stitch strategies that affect the forward graph's inputs or outputs,
   * `stitch()` should also amend all call sites of the forward graph as
   * appropriate. Conversely, for the backward graphs, it is assumed there are
   * no call sites as it's anticipated this method is called before parents of
   * the backward graph exist.
   *
   * NOTE: This method may modify the forward graph, backward graph, or any
   *    graphs that call these graphs, depending on the method. It also may
   *    raise a `popart::error` if it is unable to stitch an index.
   *
   * \param ir The IR in the context of which this transformation is applied.
   * \param fwdGraphId The ID of the subgraph to differentiate.
   * \param bwdGraphInfo The data structure describing the backward graph.
   * \param stitchStrategy The method by which to stitch any autodiff result for
   *     graphs that are directly or indirectly called by the graph.
   * \param stitchIndices If provided, backward graph input indices not in this
   *     list must be ignored and backward graph input indices in this list
   *     must be stitched (or an exception raised). If not set, it is up to the
   *     stitcher to decide what indices to stitch.
   *
   * \return An updated `BwdGraphInfo` data structure (with some
   *     `expectedInputs` removed).
   *
   * \throws popart::error if unable to stitch an index.
   **/
  virtual BwdGraphInfo
  stitch(Ir &ir,
         const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         AutodiffStitchStrategy stitchStrategy,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices);

  /**
   * Get the ID of the autodiff transform.
   */
  virtual std::size_t getId() const override { return id(); }

  /**
   * Get the name of the autodiff transform.
   */
  virtual std::string getName() const override { return "Autodiff"; }

private:
  // Helper class to create stitchers (add a setter if it helps with testing).
  std::unique_ptr<StitcherFactory> stitcherFactory;
};

} // namespace popart

#endif
