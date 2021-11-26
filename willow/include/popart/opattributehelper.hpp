// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ATTRIBUTEHELPER_HPP
#define GUARD_NEURALNET_ATTRIBUTEHELPER_HPP

#include <map>
#include <set>
#include <tuple>
#include <popart/op.hpp>

namespace popart {

class Ir;
class SessionOptions;

/**
 *  Helper class to set Op's placement attributes by inheriting them from
 * other ops in the graph. Attributes that are set include:
 * - Execution context
 * - Pipeline stage
 * - Execution phase
 * - Virtual graph ID
 * - Batch serial phase (optional)
 *
 * The most common use case is when a transform inserts a new operation into
 * the graph, and there isn't a single Op available in the graph from which
 * the above attributes could be inherited unambiguously.
 */
class InheritOpAttributeHelper {
public:
  /**
   * Integer used to steer sorting attributes by priority
   */
  using InfoSortPriority = int;

  /**
   * Enum describing the relation of a connected Op to the current Op
   */
  enum ConnectedOpRelation {
    /// The op from which to inherit is an upstream producer of an input to the
    /// current Op
    Producer = 0,
    /// The op from which to inherit is a downstream consumer of an output to
    /// the current Op
    Consumer,
    /// The number of values
    N
  };

  /**
   * Struct to sort traversed tensors by priority
   */
  struct TraversalTensorInfo {
    ConnectedOpRelation relation;
    int64_t distance;

    int64_t sortValue() const {
      // Short distance before long distance, producer before consumer
      return static_cast<int64_t>(relation) +
             static_cast<int64_t>(ConnectedOpRelation::N) * distance;
    }

    bool operator<(const TraversalTensorInfo &o) const {
      return sortValue() < o.sortValue();
    };
  };

  /**
   * Creates a helper to inherit and set attributes on the Op
   * \param op_                    Op to set attributes on
   * \param inheritSerializations_ If true, inherit the batch serial phase
   *                               attribute where available
   * \param aliasModel_            Model of all tensor aliases in all graphs
   */
  InheritOpAttributeHelper(Op *op_,
                           bool inheritSerializations_,
                           AliasModel &aliasModel_);

  /**
   * Applies a helper to inherit and set attributes on the Op
   * \param op                     Op to set attributes on
   * \param inheritSerializations  If true, inherit the batch serial phase
   *                               attribute where available
   * \param aliasModel             Model of all tensor aliases in all graphs
   */
  static void apply(Op *op, bool inheritSerializations, AliasModel &aliasModel);

private:
  /**
   * Iterate over the Op's inputs and outputs to collect starting points for
   * breadth-first graph traversal
   * \return Vector of tensors to traverse
   */
  std::vector<Tensor *> getTraversalStartTensors();

  /**
   * Traverse the graphs, tensors and operations before and after the Op to find
   * the required attributes (\c VGraphId, \c PipelineStage, \c ExecutionPhase,
   * \c ExecutionContext, \c BatchSerializedPhase)
   */
  void traverseGraphs();

  /**
   * Check if graph traversal can be stopped early
   * (before visiting all connected tensors in all graphs exhaustively)
   * \param graphChanged  True if the current traversal step changes the graph
   *                      (leaves or enters subgraph)
   * \param distance      Distance in number of traversed tensors between the Op
   *                      input/output tensors and the current tensor
   * \return              True if traversal can be stopped after having
   *                      processed all tensors with the current distance
   */
  bool canStopTraversal(bool graphChanged, int distance);

  /**
   * Finds the best set element for attribute inheritance
   * \tparam T              Attribute type that can be inherited from other Ops
   * \param preferredOpIds  OpIds that were used for inheriting other attributes
   *                        previously. The algorithm will add the OpIds that
   *                        correspond to the best selected attribute value
   *                        to the preferredOpIds.
   * \param set             The set of possible attribute values to inherit
   * \return                Iterator to the best fitting element
   */
  template <typename T>
  typename std::set<
      std::tuple<TraversalTensorInfo, InfoSortPriority, T, OpId>>::iterator
  getBestSetElement(
      std::set<OpId> &preferredOpIds,
      std::set<std::tuple<TraversalTensorInfo, InfoSortPriority, T, OpId>> set,
      bool allowNewOpId);

  /**
   * Validate and set the attributes that were found on the Op
   */
  void setAttributes();

  /**
   * Scans if the current operation modifies any weights. This modification
   * may occur through an inplace aliasing of the weight, e.g. with MatMul
   * serialization. Therefore we have to search through the aliasing chains
   * to find any directly (or through an alias) modified weights, and ensure
   * the modifying op is placed on the virtual graph that owns the weight.
   * \return The required VGraphId if it exists
   */
  OptionalVGraphId getRequiredVirtualGraphId();

  /**
   * Add found attributes from an Op to the attribute maps
   * \param visitedOp     Op from which to read the attributes
   * \param visitedTensor Tensor from which the Op was visited
   * \param info          Traversal information describing the path from the Op
   *                      on which attributes should be set to the visitedOp.
   * \param isInput       If the tensor from which the visitedOp was reached is
   *                      an input or output to the visitedOp.
   */
  void addAttributeEntry(Op *visitedOp,
                         Tensor *visitedTensor,
                         const TraversalTensorInfo &info,
                         bool isInput);

  /**
   * Decide when to stop searching for execution contexts
   * \param graphChanged True if the current traversal step changes the graph
   *                     (leaves or enters subgraph)
   * \param distance     Distance in number of traversed tensors between the Op
   *                     input/output tensors and the current tensor
   * \return             true when the search can be stopped
   */
  bool stopExecutionContextSearch(bool graphChanged, int distance);

  /**
   * Decide when to stop searching for virtual graph IDs
   * \param graphChanged True if the current traversal step changes the graph
   *                     (leaves or enters subgraph)
   * \param distance     Distance in number of traversed tensors between the Op
   *                     input/output tensors and the current tensor
   * \return             true when the search can be stopped
   */
  bool stopVGIDSearch(bool graphChanged, int distance);

  /**
   * Decide when to stop searching for pipeline stages
   * \param graphChanged True if the current traversal step changes the graph
   *                     (leaves or enters subgraph)
   * \param distance     Distance in number of traversed tensors between the Op
   *                     input/output tensors and the current tensor
   * \return             true when the search can be stopped
   */
  bool stopPipelineStageSearch(bool graphChanged, int distance);

  /**
   * Decide when to stop searching for execution phases
   * \param graphChanged True if the current traversal step changes the graph
   *                     (leaves or enters subgraph)
   * \param distance     Distance in number of traversed tensors between the Op
   *                     input/output tensors and the current tensor
   * \return             true when the search can be stopped
   */
  bool stopExecutionPhaseSearch(bool graphChanged, int distance);

  /**
   * Decide when to stop searching for batch serialized phases
   * \param graphChanged True if the current traversal step changes the graph
   *                     (leaves or enters subgraph)
   * \param distance     Distance in number of traversed tensors between the Op
   *                     input/output tensors and the current tensor
   * \return             true when the search can be stopped
   */
  bool stopBatchSerializedPhaseSearch(bool graphChanged, int distance);

  Op *op;

  AliasModel &aliasModel;

  // Batch serialization phase inherited
  bool inheritSerializations;

  const Ir &ir;
  const SessionOptions &opts;

  // Virtual graphs enabled
  bool vgraphed;

  // Pipelining enabled
  bool pipelined;

  // Execution phases enabled
  bool executionphased;

  // Map to track if we are visiting in forward or backward direction
  std::map<Tensor *, TraversalTensorInfo, PTensorCmp> tensorRelationMap;

  // Maps for the attributes that can be inherited
  std::set<std::tuple<TraversalTensorInfo, InfoSortPriority, VGraphId, OpId>>
      vgidSet;
  std::set<
      std::tuple<TraversalTensorInfo, InfoSortPriority, PipelineStage, OpId>>
      pipelineStageSet;
  std::set<
      std::tuple<TraversalTensorInfo, InfoSortPriority, ExecutionPhase, OpId>>
      executionPhaseSet;
  std::set<
      std::tuple<TraversalTensorInfo, InfoSortPriority, ExecutionContext, OpId>>
      executionContextSet;
  std::set<std::tuple<TraversalTensorInfo,
                      InfoSortPriority,
                      BatchSerializedPhase,
                      OpId>>
      batchSerializedPhaseSet;
};

std::ostream &
operator<<(std::ostream &os,
           InheritOpAttributeHelper::ConnectedOpRelation opRelation);
std::ostream &
operator<<(std::ostream &os,
           const InheritOpAttributeHelper::TraversalTensorInfo &info);

} // namespace popart

#endif
