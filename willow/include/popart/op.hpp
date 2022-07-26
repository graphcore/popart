
// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <poprithms/memory/inplace/proposal.hpp>
#include <popart/attributes.hpp>
#include <popart/basicoptionals.hpp>
#include <popart/bwdgraphinfo.hpp>
#include <popart/names.hpp>
#include <popart/opdebuginfo.hpp>
#include <popart/scope.hpp>
#include <popart/subgraph/subgraphnames.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensorlocation.hpp>
#include <popart/vertex.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

class AliasModel;
class CommGroup;
class Graph;
class Ir;
class Pattern;
class ReplicaEqualAnalysisProxy;
class Tensor;
class TensorIndexMap;
class any;

/// Define the type of recomputation.
enum class RecomputeType {

  /// Default value if RecomputeType has not been set.
  Undefined = 0,

  /// Do not recompute. Outputs from the op are kept from the forward pass.
  Checkpoint,

  /// Recompute operation.
  Recompute,

  /**
   * For explicit recomputation, this marks a cloned operation that had
   * RecomputeType::Recompute set. After cloning, the original op is changed to
   * RecomputeType::Checkpoint, and the cloned op is changed to Recomputed.
   */
  Recomputed,

};

/// Define the type of the execution context.
enum class ExecutionContext {

  /// Run the forward and backward passes (Default).
  Normal = 0,

  /**
   * Used to run the AccumulateOps after the gradient accumulation loop
   * completes.
   */
  AccumulateOuterFragment,

  /// Used to transfer weights from host to device.
  WeightsFromHostFragment,

  /// Used to download weights from the device to the host.
  WeightsToHostFragment,

  /// Used to stream the optimizer state from the host.
  OptimizerFromHostFragment,

  /// Program fragment used for subgraph-specific operations.
  Subgraph,
};

/**
 * Define the reduction operation to use over a sequence of tensors.
 *
 * The two use-cases for this enum type are:
 *    * denoting how to reduce individual losses produced by a LossOp over a
 *      minibatch (specified by the LossOp `reduction` parameter)
 *    * denoting how to reduce weight gradients over a number of replicas when
 *      gradient accumulation is enabled (specified by the global session option
 *      SessionOptions::accumulationAndReplicationReductionType).
 */
enum class ReductionType {
  /// Sum the input values and do not scale the output (Default).
  Sum = 0,
  /// Take the mean of the input values.
  Mean,
  /**
   * Do not reduce the input values. Keep them stacked into a single
   * tensor. So values \f$t_1, ..., t_k\f$ get collected into a tensor
   * \f$[t_1, ..., t_k]\f$.
   */
  NoReduction,
  /// The number of ReductionType values.
  N
};

std::ostream &operator<<(std::ostream &, const RecomputeType &);
std::ostream &operator<<(std::ostream &, const ExecutionContext &);
std::ostream &operator<<(std::ostream &, const ReductionType &);

class OpSerialiserBase;
class ShardingPlan;

/**
 * Define the relationship between the input tensors of a gradient operation and
 * the corresponding non-gradient operation.
 */
// design note: it's not possible for an input to a grad op to NOT be directly
// related to the corresponding non-grad op.
enum class GradOpInType {
  /**
   * Indicates that the input tensor to the gradient operation is an input
   * tensor of the non-gradient operation (Default).
   */
  In = 0,

  /**
   * Indicates that the input tensor to the gradient operation is an output
   * tensor of the non-gradient operation.
   */
  Out,

  /**
   * Indicates that the input tensor to the gradient operation is an output
   * gradient tensor of the non-gradient operation.
   */
  GradOut,
};

/**
 * Class that represents the mapping between the indices of the input tensors
 * to the gradient operation and the indices of these same tensors in the
 * non-gradient operation.
 */
class GradInOutMapper {
public:
  /**
   * Constructor for the GradInOutMapper class.
   *
   * \param iGrad_ The index of the input tensor to the gradient operation.
   * \param iNonGrad_ The index of the gradient operation input tensor as it
   *      is indexed in the non-gradient operation.
   * \param GradOpInType The type of the input tensor to the gradient operation.
   */
  GradInOutMapper(InIndex iGrad_, int iNonGrad_, GradOpInType);
  // Input index to a gradient operation.
  InIndex iGrad;
  // "input/output/gradient-of-output" index to
  // corresponding non-grad op,
  int iNonGrad;
  // where "input/output/gradient-of-output" above is
  GradOpInType type;

  /**
   * Check if the current GradInOutMapper object is equal to another
   * GradInOutMapper object.
   *
   * \param rhs A GradInOutMapper object to be compared to the current object.
   *
   * \returns `true` if objects are equal, `false` otherwise.
   */
  bool operator==(const GradInOutMapper &rhs) const;
};

// Map and set classes for `popart::Op *` for deterministic iteration order.
template <typename T> using OpMap      = std::map<Op *, T, POpCmp>;
template <typename T> using ConstOpMap = std::map<const Op *, T, POpCmp>;
using OpSet                            = std::set<Op *, POpCmp>;
using ConstOpSet                       = std::set<const Op *, POpCmp>;

// clang-off
/**
 * Parent class for the concrete \c Op implementations.
 *
 * The \c poplar implementation which the op represents can be found in the
 * corresponding popx::Opx class, and will be lowered to \c poplar.
 *
 * \sa [Custom ops in the PopART User
 * Guide](https://docs.graphcore.ai/projects/popart-user-guide/en/latest/custom_ops.html).
 */
// clang-on
class Op : public Vertex {
public:
  // We use pointers to TensorIndexMaps for PIMPL reasons.
  // Note that we cannot initialise these with {nullptr} on gcc.
  // They are initialised in the op constuctors

  // A map of indices for op input tensors.
  // Input tensors enter an op at specific indices and output tensors leave the
  // op at specific indices. Each index has only one associated tensor, but each
  // tensor can be associated with more than one index.
  std::unique_ptr<TensorIndexMap> input;

  // A map of indices for op output tensors.
  // Input tensors enter an op at specific indices and output tensors leave the
  // op at specific indices. Each index has only one associated tensor, but each
  // tensor can be associated with more than one index.
  std::unique_ptr<TensorIndexMap> output;

  // The unique identifier of the op that will always be set in Op::Op.
  OpId id{-1};

  // The operation type, domain and version.
  // An operator is identified by a three-part identifier formatted as
  // `domain.op_type:op_version`, for example `com.acme.FastConv:3`.
  // Nodes in graphs always refer to operators by this three-part identifier.
  OperatorIdentifier opid;

  // Indicate that the op is pruneable (`true`) or not (`false`).
  // All ops are pruneable by default.
  bool pruneable = true;

  /// Structure to capture the settings for the op.
  struct Settings {

    /**
     * Constructor for the Settings structure.
     *
     * \param graph_ The graph the op belongs to.
     * \param name_ The name of the op.
     */
    Settings(Graph &graph_, const std::string &name_)
        : graph(graph_), name(name_) {
      DebugInfo di({name}, "popartbuilder");
      debugInfoId = di.getId();
    }

    /**
     * Constructor for the Settings structure.
     *
     * \param graph_ The graph the op belongs to.
     * \param name_ The name of the op.
     * \param scope_ The scope of the op.
     */
    Settings(Graph &graph_, const std::string &name_, const Scope &scope_)
        : graph(graph_), name(name_), scope(scope_) {
      DebugInfo di({name}, "popartbuilder");
      debugInfoId = di.getId();
    }

    /**
     * Constructor for the Settings structure.
     *
     * \param graph_ The graph the op belongs to.
     * \param name_ The name of the op.
     * \param scope_ The scope of the op.
     * \param parentId_ The ID of the debug info.
     */
    Settings(Graph &graph_,
             const std::string &name_,
             const Scope &scope_,
             const uint64_t parentId_)
        : graph(graph_), name(name_), scope(scope_), debugInfoId(parentId_) {}

    /**
     * Constructor for the Settings structure.
     *
     * \param graph_ The main graph.
     * \param name_ The name of the op.
     * \param parentId_ The ID of the debug info.
     */
    Settings(Graph &graph_, const std::string &name_, const uint64_t parentId_)
        : graph(graph_), name(name_), debugInfoId(parentId_) {}

    /// Destructor for the Settings structure.
    virtual ~Settings() = default;

    Settings(const Settings &) = default;

    /**
     * Create a copy of the current settings with a new name.
     *
     * \param new_name The name of the new settings.
     * \returns A copy of the current settings with the new name.
     */
    Settings copy(const std::string &new_name) {
      Settings s = *this;
      s.name     = new_name;
      return s;
    }

    std::reference_wrapper<Graph> graph;

    std::string name = "";

    // The scope that this op has been assigned to.
    Scope scope;
    // The default recompute type for this op.
    RecomputeType recomputeType = RecomputeType::Undefined;
    // (Optional) The tensor location.
    OptionalTensorLocation tensorLocation;

    // optional inplace priorities, to take precedence over the default
    // priorities. A negative priority gurarantees no inplacing
    // This should really be a map with "OperatorIdentifier" keys, see T6783
    std::vector<std::tuple<std::string, float>> inplacePriorityVeto;

    // A set of patterns which should not be applied to this op.
    std::unordered_set<std::string> excludePatterns;

    // (Optional) The virtual graph this op has been assigned to.
    OptionalVGraphId vgraphId;

    // (Optional) The pipeline stage this op has been assigned to.
    OptionalPipelineStage pipelineStage;

    // (Optional) The execution phase this op has been assigned to.
    OptionalExecutionPhase executionPhase;

    // (Optional) The batch serialized phase this op has been assigned to.
    OptionalBatchSerializedPhase batchSerializedPhase;

    // The desired stochastic rounding behaviour, if set.
    OptionalStochasticRoundingMethod stochasticRoundingMethod;

    // If the OP should be placed on I/O tiles instead of regular tiles
    TileSet tileSet{TileSet::Compute};

    // If the OP needs to run in a special fragment,
    // such as gradient accumulation
    ExecutionContext executionContext{ExecutionContext::Normal};

    // Tensor layout mapping should be inferred "to" tensor <- "from" tensor
    std::map<InIndex, InIndex> inferTensorMappingToFrom;

    // all ops will be topologically sorted "as close to" the order of
    // priority (highest to lowest) while still resulting in a valid
    // topological ordering.
    // default : 0.0
    double schedulePriority{0.0};

    // Extra attributes to differentiate ops for outlining
    // ops with different outline attributes are not outlined together
    std::map<std::string, std::string> extraOutlineAttributes;

    // The debug info ID of the parent debug info for this op.
    uint64_t debugInfoId{0};

    // To flag an op as being part of the optimizer
    bool optimizerOp{false};

    // To flag an op as being part of gradient clipping
    bool gradientClippingOp{false};

    /**
     * Append the optional attributes to the Settings structure depending on
     * whether the attribute has been set in the ONNX model.
     *
     * \param attributes The attributes to be added to the Settings structure.
     */
    virtual void setFromAttributes(const Attributes &attributes);

    /**
     * Get the IR associated with the main graph.
     * \returns The IR associated with the main graph.
     */
    Ir &getIr() const;
  };

  // the op settings
  Settings settings;

  // The op debug information
  OpDebugInfo debugInfo;

  /**
   * Get the settings associated with the op.
   * \returns The op settings.
   */
  Settings &getSettings() { return settings; }

  /**
   * Get the settings associated with the op.
   * \returns The op settings.
   */
  const Settings &getSettings() const { return settings; }

  /**
   * Return suitable settings for an op inserted before the input to an
   * existing op.
   * \param InIndex The input index before which the op is inserted.
   * \returns The settings for the op inserted before the input index.
   */
  virtual Settings getInSettings(InIndex) const;

  /**
   * Return suitable settings for an op inserted after the output to an
   * existing op.
   * \param OutIndex The output index after which the op is inserted.
   * \returns The settings for the op inserted after the output index.
   */
  virtual Settings getOutSettings(OutIndex) const;

  /**
   * Adjust the settings to be suitable as input at the input index.
   * \param InIndex The input index where the settings are to be applied.
   * \param Settings The settings to be adjusted.
   * \returns Adjusted settings suitable for input at the input index.
   */
  Settings adjustInSettings(InIndex, Op::Settings) const;

  /**
   * Adjust the settings to be suitable as output at an output index.
   * \param OutIndex The output index where the settings are to be applied.
   * \param Settings The settings to be adjusted.
   * \returns Adjusted settings suitable for output at the output index.
   */
  Settings adjustOutSettings(InIndex, Op::Settings) const;

  /**
   * Get the ID of the optional virtual graph.
   * \returns The ID of the optional virtual graph.
   */
  const OptionalVGraphId getOptionalVGraphId() const;

  /**
   * Get the ID of the  virtual graph.
   * \returns The ID of the virtual graph.
   */
  VGraphId getVirtualGraphId() const;

  /**
   * Get virtual graph ID and tile set associated with an input index.
   * \param InIndex The input index.
   * \returns The virtual graph ID and tile set at the input index.
   */
  VGraphIdAndTileSet getIntrospectionInVirtualGraphId(InIndex) const;

  /**
   * Get virtual graph ID and tile set associated with an output index.
   * \param OutIndex The output index.
   * \returns The virtual graph ID and tile set at the output index.
   */
  VGraphIdAndTileSet getIntrospectionOutVirtualGraphId(OutIndex) const;

  /**
   * Get virtual graph ID and tile set associated with an input index.
   * \param InIndex The input index.
   * \param visited The set of labels associated with this operator to
   *       distinguish it from other operators in the virtual graph.
   * \returns The virtual graph ID and tile set at the input index.
   */
  virtual VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex, std::set<OpId> &visited) const;

  /**
   * Get virtual graph ID and tile set associated with an output index.
   * \param OutIndex The output index.
   * \param visited The set of labels associated with this operator to
   *       distinguish it from other operators in the virtual graph.
   * \returns The virtual graph ID and tile set at the output index.
   */
  virtual VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex, std::set<OpId> &visited) const;

  /**
   * Set a virtual graph ID for the op.
   * \param OptionalVGraphId The ID of the virtual graph to set on this op.
   */
  void setVirtualGraphId(const OptionalVGraphId);

  /**
   * Check if the op has a virtual graph ID set.
   * \returns `true` if the op has a virtual graph ID set, `false` otherwise.
   */
  bool hasVirtualGraphId() const;

  /**
   * Set a pipeline stage for the op.
   * \param OptionalPipelineStage The pipeline stage to be set for the op.
   */
  void setPipelineStage(OptionalPipelineStage);

  /**
   * Check if the op has a pipeline stage set.
   * \returns `true` if the op has a pipeline stage set, `false` otherwise.
   */
  bool hasPipelineStage() const;

  /**
   * Get the pipeline stage that has been set for the op.
   * \returns The pipeline stage that has been set for the op.
   */
  PipelineStage getPipelineStage() const;

  /**
   * Get the optional pipeline stage.
   * \returns The optional pipeline stage that has been set for the op.
   */
  OptionalPipelineStage getOptionalPipelineStage() const;

  /**
   * Get the optional execution phase.
   * \returns The optional execution phase that has been set for the op.
   */
  const OptionalExecutionPhase getOptionalExecutionPhase() const;

  /**
   * Get the execution phase that has been set for the op.
   * \returns The execution phase that has been set for the op.
   */
  virtual ExecutionPhase getExecutionPhase() const;

  /**
   * Set the execution phase for the op.
   * \param OptionalExecutionPhase The execution phase to be set for the op.
   */
  void setExecutionPhase(const OptionalExecutionPhase);

  /**
   * Check if the op has an execution phase set.
   * \returns `true` if the op has a execution phase set, `false` otherwise.
   */
  bool hasExecutionPhase() const;

  /**
   * Get the optional batch serialized phase.
   * \returns The optional batch serialized phase that has been set for the op.
   */
  const OptionalBatchSerializedPhase getOptionalBatchSerializedPhase() const;

  /**
   * Get the batch serialized phase.
   * \returns The batch serialized phase that has been set for the op.
   */
  virtual BatchSerializedPhase getBatchSerializedPhase() const;

  /**
   * Set the batch serialized phase.
   * \param OptionalBatchSerializedPhase The batch serialized phase to be set
   *       for the op.
   */
  void setBatchSerializedPhase(const OptionalBatchSerializedPhase);

  /**
   * Check if the op has a batch serialization phase set.
   * \returns `true` if the op has a batch serialization phase set, otherwise
   *       `false`.
   */
  bool hasBatchSerializedPhase() const;

  /**
   * Get the optional stochastic rounding method.
   * \returns The optional stochastic rounding method that has been set for the
   *       op.
   */
  const OptionalStochasticRoundingMethod
  getOptionalStochasticRoundingMethod() const;

  /**
   * Get the stochastic rounding method.
   * \returns The stochastic rounding method that has been set for the
   *       op.
   */
  virtual StochasticRoundingMethod getStochasticRoundingMethod() const;

  /**
   * Set the optional stochastic rounding method.
   * \param OptionalStochasticRoundingMethod The optional stochastic rounding
   *       method to be set for the op.
   */
  void setStochasticRoundingMethod(const OptionalStochasticRoundingMethod);

  /**
   * Check if the op has a stochastic rounding method set.
   * \returns `true` if the op has a stochastic rounding method set, otherwise
   *       `false`.
   */
  bool hasStochasticRoundingMethod() const;

  /**
   * Check if the op is excluded from a pattern.
   * \returns `true` if the op is excluded from a pattern, `false` otherwise.
   */
  bool isExcludedFromPattern(const Pattern *) const;

  /**
   * Get the batch axis for the input index.
   * \returns The batch axis for the input index.
   */
  virtual int getInBatchAxis(InIndex) const { return 0; }

  /**
   * Get the batch axis for the output index.
   * \returns The batch axis for the output index.
   */
  virtual int getOutBatchAxis(OutIndex) const { return 0; }

  /**
   * Helper function to set an op's placement attributes by inheriting them from
   * other ops in the graph. The attributes that are set include:
   * - Execution context.
   * - Pipeline stage.
   * - Execution phase.
   * - Virtual graph ID.
   * - Batch serial phase (optional).
   *
   * \param inheritSerializations The indicator to enable or disable the batch
   *      serialization phase. `true` enables the batch serialization phase and
   *      `false` disables it.
   * \param aliasModel An AliasModel object containing alias info for this op's
   *      graph.
   */
  void inheritPlacementAttributes(bool inheritSerializations,
                                  AliasModel &aliasModel);

  /**
   * Get the IR associated with the op.
   * \returns The IR associated with the op.
   */
  Ir &getIr();

  /**
   * Get the IR associated with the op.
   * \returns The IR associated with the op.
   */
  const Ir &getIr() const;

  /**
   * Get the graph associated with the op.
   * \returns The graph associated with the op.
   */
  Graph &getGraph() { return settings.graph.get(); }

  /**
   * Get the graph associated with the op.
   * \returns The graph associated with the op.
   */
  const Graph &getGraph() const { return settings.graph.get(); }

  /**
   * Get the scope associated with the op.
   * \returns The scope associated with the op.
   */
  const Scope &getScope() const { return settings.scope; }

  /**
   * Get the scope associated with the op.
   * \returns The scope associated with the op.
   */
  void setScope(const Scope &scope) { settings.scope = scope; }

  /**
   * Get the name of the op.
   * \returns The name of the op.
   */
  const std::string &getName() const { return settings.name; }

  /**
   * Get the name of the op.
   * \returns The name of the op.
   */
  void setName(const std::string &name) { settings.name = name; }

  /**
   * Get the debug info of the op.
   * \returns The debug info for the op.
   */
  const OpDebugInfo &getDebugInfo() const { return debugInfo; }

  /**
   * Checks if the op is a norm op.
   * \returns `true` if the op is a norm op, `false` otherwise.
   */
  virtual bool isNorm() const;

  /**
   * Checks if the op is an element-wise unary op.
   * \returns `true` if the op is an element-wise unary op, `false` otherwise.
   */
  bool isElementWiseUnary() const;

  // Methods used by patterns to determine if an op can be replaced by another
  // op

  /**
   * Check if the op can be replaced by the identity op.
   * \returns `true` if the op and be replaced by the identity op, `false`
   *       otherwise.
   */
  virtual bool canBeReplacedByIdentity() const;

public:
  /**
   * Constructor of the \c Op class.
   *
   * \param _opid The operator identifier specifying domain:type:version,
   *      minimum and maximum number of input tensors and number of output
   *      tensors.
   * \param settings The general op settings such as graph, name and scope.
   */
  Op(const OperatorIdentifier &_opid, const Op::Settings &settings);

  /// Copy constructor. \note This does NOT copy input and output.
  Op(const Op &);
  Op &operator=(const Op &) = delete;
  // A c++ aside: the rule-of-3 says that it's good
  // practise to have an explicit destructor,
  // given that there is an explict copy con.
  // But not really nec. as Vertex has a virtual destructor.

  /// Destructor.
  virtual ~Op();

  /// Return the op ID.
  std::string str() const final;

  /// Return the op name that is used for debug and profiling.
  std::string debugName() const;

  /**
   * Create an ActGrad (output) tensor and connect it to this op's output.
   * \param OutIndex The output index that the output tensor should be connected
   *       to.
   * \param TensorId The tensor ID of the tensor to be converted to an output
   *       tensor.
   */
  void createAndConnectOutTensor(OutIndex, TensorId);

  /**
   * Append this op to a stream.
   * \param ss The stream to append the op to.
   */
  void append(std::stringstream &ss) const;

  /**
   * Convert this op to JSON format and append it to a stream.
   * \param ss The stream to append the JSON-serialised op to.
   */
  void toJSON(std::stringstream &ss) const;

  /**
   * Return the total memory of used by all output tensors.
   */
  // We might want a cycle counter too for more sophisticated recomputation
  int64_t memOfOutputs() const;

  /**
   * Return the input indices of all optional inputs to the
   * op.
   */
  virtual std::set<InIndex> optionalInputs() const { return {}; }

  /**
   * Connect a tensor to an input index.
   * This method updates the input and updates consumers of the tensor with the
   * tensor ID.
   * \param InIndex The input index to connect the tensor to.
   * \param TensorId The tensor ID of the tensor to connect.
   */
  void defaultConnectInTensor(InIndex, TensorId);

  /**
   * Connect existing tensor to input index.
   * \param index The input index at which to connect the tensor.
   * \param tensorId The ID of the existing tensor.
   */
  virtual void connectInTensor(InIndex index, TensorId tensorId);

  /**
   * Connect an existing tensor to an index with the source virtual graph.
   * \param inIndex The input index at which to connect the tensor.
   * \param tensorId The ID of the existing tensor.
   * \param vgid The virtual graph on which the existing tensor resides.
   */
  virtual void
  connectInTensor(InIndex inIndex, TensorId tensorId, VGraphId vgid);

  /**
   * Connect an existing tensor at an index with the source virtual graph.
   *
   * Dispatcher to resolve issues with templated inheritance overloads.
   * This will automatically derive the virtual graph ID of the input when
   * required.
   *
   * \param inIndex The input index at which to connect the tensor.
   * \param tensorId The ID of the existing tensor.
   */
  void connectInTensorDispatch(InIndex inIndex, TensorId tensorId);

  /**
   * Connects the input tensor analogously to another op.
   * This is useful when cloning graphs or ops, because it avoids having to
   * check if the op requires special considerations when connecting inputs.
   *
   * IpuCopyOp is currently the only op where this applies, since a source
   * virtual graph has to be specified when connecting it otherwise:
   *
   * \code
   *   void connectInTensor(InIndex, TensorId, uint64_t sourceIpu);
   * \endcode
   * \param other An op of the same type as the current op, from which to
   *      copy how the tensor at the corresponding index should be
   *      connected.
   * \param index The input index to connect.
   * \param tenId The ID of the tensor to connect.
   */
  void connectInTensorLike(const Op *other, InIndex index, TensorId tenId);

  /**
   * Connect existing tensor to output index.
   * \param index The output index at which to connect the tensor.
   * \param tensorId The ID of the existing tensor.
   */
  void connectOutTensor(OutIndex, TensorId);

  /**
   * Disconnect an input tensor from the op.
   * \param tensor The tensor to disconnect.
   */
  void disconnectInTensor(Tensor *tensor);

  /**
   * Disconnect an input tensor from the op at a specific input index.
   * \param tensor The tensor to disconnect.
   * \param InIndex The index of the input tensor in the op.
   */
  virtual void disconnectInTensor(InIndex, Tensor *tensor);

  /**
   * Disconnect an input tensor from the input index.
   * \param InIndex The input index to disconnect the tensor from.
   */
  void disconnectInTensor(InIndex);

  /**
   * Disconnect an output tensor from the op.
   * \param tensor The tensor to disconnect.
   */
  void disconnectOutTensor(Tensor *tensor);

  /// Disconnect all input tensors from the op.
  void disconnectAllInputs();

  /// Disconnect all output tensors from the op.
  void disconnectAllOutputs();

  /// Return the op name.
  const std::string &name() const;

  /**
   * Set the shape and type of the arguments to the op.
   * This MUST set the type and shape information for all the output
   * TensorInfo objects.
   */
  virtual void setup();

  /**
   * Finalize DebugInfo.
   * This method is called once after Ir::prepare() has completed.
   */
  void finalizeDebugInfo();

  /**
   * Set information about the gradient graphs for this op's called subgraphs.
   * If the op has called subgraphs, then this method will get called prior to
   * `getGradOps()` to provide the op with the information it needs to call the
   * grad version of the called subgraphs.
   * \param calledGraphsGradInfo The mapping between the forward graph and
   *       information on the gradient graph.
   */
  virtual void
  setCalledSubgraphGradInfo(const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);

  /**
   * Determine the corresponding grad op for each op in the forward graph to
   * automatically generate the backward pass.
   *
   * There can be a separate gradient op for each input or a single gradient
   * op that generates gradients for all inputs.
   *
   * The mapping from the index of each output tensor of the gradient op to the
   * index of each input tensor of the non-grad op is configured using the
   * gradOutToNonGradIn() method that should be overridden in the grad op
   * definitions.
   *
   * Throws an error if this op is already a gradient op.
   */
  virtual std::vector<std::unique_ptr<Op>> getGradOps();

  /**
   * Return the variants of this op (if any) which can
   * modify / alias the inputs at the given indices.
   * This function doesn't check for anchor violations
   * or topological order violations. When there are several ops,
   * they should be returned in descending order of preference
   * If the op can be replaced by an in-place variant of itself, this method
   * should be overridden to return a vector of <OperatorIdentifier, float>
   * tuples in descending order of preference.
   */
  virtual std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const;

  /**
   * Instantiate a particular in-place variant of the op with a specified
   * OperatorIdentifier from the vector returned by inplacePriorityDefault().
   * \param OperatorIdentifier The operator identifier of the op to be
   *       instantiated.
   * \returns An instance of the required op.
   */
  virtual std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const;

  // clang-off
  /**
   * For certain tasks which involve analysing how tensors alias each other,
   * such as inplacing, a
   * [poprithms::memory::inplace::Graph](https://github.com/graphcore/poprithms)
   * that corresponds to this op's graph is constructed. The Poprithms graph
   * can then be queried for aliasing information, and can have algorithms run
   * on it.
   *
   * To construct the Poprithms graph, each PopART op defines what its
   * Poprithms equivalent ops are. This method inserts this op's
   * poprithms::memory::inplace::Op equivalents into the [Poprithms
   * Graph](https://github.com/graphcore/poprithms), which is the container \a
   * popAliaser.
   *
   * \pre All input tensors of this op have mappings in \p aliasModel
   *     before the call to \p aliasModel.
   * \post All output tensors of this op have mappings in \p aliasModel
   *     after to the call to \p aliasModel.
   * \sa AliasModel.
   * \param aliasModel The mapping between this op's (PopART) graph and
   *     the Poprithms graph.
   */
  // clang-on
  virtual void growAliasModel(AliasModel &aliasModel) const;

  /**
   * Translate a PopART inplacing proposal.
   * This replaces an outplace op with an inplace op of type \p inplaceId,
   * into an AliasModel equivalent.
   *
   * \param aliasModel The mapping between this op's (PopART) graph and
   *      the Poprithms graph.
   * \param 2 The operator identifier to translate to the
   *      AliasModel equivalent.
   *
   * \return A tuple where the first element corresponds to an alias gate in the
   * AliasModel and the second element is a input index.
   *
   * This method is defined as a void method which sets a value passed by
   * reference, as opposed to a getter method, so that no Poprithms headers need
   * to be included in this file.
   */
  virtual poprithms::memory::inplace::Proposal
  mapInplaceProposal(const AliasModel &aliasModel, OperatorIdentifier) const;

protected:
  /**
   * This method is a possible implementation of the virtual method
   * growAliasModel, which can be used by ops which do not have more than 1
   * variant (that is, they have no "inplace" variants), and do no non-trivial
   * view-changing. Examples: LSTM, Conv, Matmul and SumReduce.
   */
  void growAliasModelMulti(AliasModel &) const;

  // clang-off
  /**
   * Set the proposal to open the poprithms::AliasGate which corresponds to this
   * op, at input index 0. For information on AliasGate and inplacing
   * proposals, see the [Poprithms
   * memory::inplace](https://github.com/graphcore/poprithms) project.
   */
  // clang-on
  virtual poprithms::memory::inplace::Proposal
  mapInplaceProposalGate0(const AliasModel &aliaser,
                          OperatorIdentifier opId) const;

public:
  /**
   * Return the input region which this op modifies (for inplace ops).
   * \param InIndex The input index.
   * \returns The regions which this op modifies.
   */
  virtual view::Regions modifies(InIndex) const;
  /**
   * Return the input region which this op uses.
   * \param InIndex The input index.
   * \returns The regions which this op uses.
   */
  virtual view::Regions uses(InIndex) const;

  // clang-off
  /**
   * Return the input region which the op output will alias (for inplace and
   * view-changing ops).
   * \param InIndex The input index.
   * \param OutIndex The output index.
   * \returns The regions which the output will alias.
   * \sa For more information on views, refer to the [IPU Programmer's
   * Guide](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/programming_model.html#data-variables).
   */
  // clang-on
  virtual view::Regions aliases(InIndex, OutIndex) const;

  // Map used regions of the input to/from the output (we assume the same for
  // modifies, aliases, uses)
  /**
   * Map regions of the input tensor at the input index to the regions of the
   * output tensor at the output index that these input regions alias.
   * \param InIndex The op input index.
   * \param OutIndex The op output index.
   */
  virtual view::RegMap fwdRegMap(InIndex, OutIndex) const;

  /**
   * Map regions of the output tensor at the output index to the regions of the
   * input tensor at the input index that these output regions alias.
   * \param InIndex The op input index.
   * \param OutIndex The op output index.
   */
  virtual view::RegMap bwdRegMap(InIndex, OutIndex) const;

  /**
   * Determine whether output tensors are guaranteed to have an equal value
   * across all replicas.
   * This means that they are "replica equal". The check is based on
   * information about the replica equal status of input tensors (and the same
   * for any inputs that are modified by the op).
   *
   * The default implementation sets each output tensor as being replica-equal
   * if and only if all tensor inputs are replica-equal. For modified inputs,
   * the default is to assume it is replica-equal only if there is an output
   * that is deemed replica-equal that fully aliases all elements of the input.
   * This default implementation is not correct for all ops. Ops that need a
   * specialized implementation should override this virtual function.
   *
   * \param aliasModel An alias model object.
   * \param inputMap A map that stores, for each input, whether the
   *     inputs are data-equivalent over all replicas.
   * \param proxy A helper object passed in by the replica-equal analysis.
   * \return A tuple comprising of:
   *          -# a mapping from output index to a replica-equal status with an
   *             entry for each output tensor.
   *          -# a vector of input indices for inputs that were modified by the
   *             op to a value that is not replica-equal.
   */
  virtual std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const;

  /**
   * Check if any input tensor aliases any output tensor .
   * \returns `true` if any input tensor aliases any output tensor, otherwise
   *      `false`.
   */
  bool doesAlias() const;

  /**
   * Check if this is an outplace op.
   * This means that no input tensor aliases any output tensor.
   * \returns `true` if this is an outplace op, otherwise
   *      `false`.
   */
  bool isOutplace() const { return !doesAlias(); }

  /**
   * Check that the input tensor at an input index aliases the output tensor at
   * an output index.
   * \returns `true` if the input tensor at \p inIndex aliases the output tensor
   *      at \p outIndex, `false` otherwise.
   */
  bool doesAlias(InIndex inIndex, OutIndex outIndex) const;

  /**
   * Check if op modifies a tensor at any index.
   *
   * \returns `true` if the op modifies a tensor at any index, otherwise
   *      `false`.
   */
  bool modifies() const;

  /**
   * Check if an op modifies a tensor at a specific index.
   *
   * \param in The input index to check.
   * \returns `true` if the op modifies the tensor, `false` otherwise.
   */
  bool modifiesIndex(InIndex in) const;

  /**
   * Check if an op overwrites a tensor.
   *
   * \param t The tensor to check.
   * \returns `true` if it overwrites the tensor, `false` otherwise.
   */
  bool overwritesTensor(Tensor *t) const;

  /**
   * Check if an op modifies a tensor.
   *
   * \param t The tensor to check.
   * \returns `true` if it modifies the tensor, `false` otherwise.
   */
  bool modifiesTensor(Tensor *t) const;

  // clang-off
  /**
   * Check if this is an inplace op that changes a view.
   * Examples of inplace ops that change views are:
   *   * ReshapeInplaceOp
   *   * IdentityInplaceOp
   *   * TransposeInplaceOp.
   * \sa For more information on views, refer to the [IPU Programmer's
   * Guide](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/programming_model.html#data-variables).
   * \returns `true` if this is a view changing inplace op, `false` otherwise.
   */
  // clang-on
  virtual bool isInplaceViewChange() const { return false; }

  // clang-off
  /**
   * Check if this is an outplace op that changes a view.
   * Examples of outplace ops that change views are:
   *   * ReshapeOp
   *   * IdentityOp
   *   * TransposeOp.
   * \sa For more information on views, refer to the [IPU Programmer's
   * Guide](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/programming_model.html#data-variables).
   * \returns `true` if this is a view changing outplace op, otherwise
   *      `false`.
   */
  // clang-on
  virtual bool isOutplaceViewChange() const { return false; }

  // A grad op outputs an edge-gradient tensor dT at gradOpOutIndex.
  // dT is the edge-gradient of a tensor T which was the input
  // to the grad op's non-grad partner. At what index was T the input
  // of non-grad op? If not relevant (non-grad ops) throw an error
  /**
   * Return the index in the non-grad op which has an output edge-gradient
   * tensor in the matching grad op.
   * This method throws an error if the op this is called on is not a grad op.
   * \param gradOpOutIndex The index at which the grad op has an output of an
   *       edge-gradient tensor.
   * \return The index in the non-grad op containing the input tensor
   *       corresponding to the edge-gradient tensor in the grad op output.
   */
  virtual int getNonGradInIndex(int gradOpOutIndex) const;

  // For grad ops, matching input indices to
  // corresponding IN/OUT/GRADOUT indices of
  // corresponding non-grad op.
  // throws an error if not appropriate (non-grad ops)
  /**
   * Get the mapping between input indices in the grad op (for inputs,
   * outputs and grad outputs) to the input indices in the corresponding
   * non-grad op.
   * This method throws an error if the op this is called on is not a grad op.
   * \returns The mapping between input indices in the grad op (for inputs,
   *       outputs and grad outputs) to the input indices in the corresponding
   *       non-grad op.
   */
  virtual const std::vector<GradInOutMapper> &gradInputInfo() const;

  /**
   * Get the mapping between the grad op outputs and the inputs of the
   * corresponding non-grad op.
   * This method throws an error if the op this is called on is not a grad op.
   */
  virtual const std::map<int, int> &gradOutToNonGradIn() const;

  // return a copy of self, similar to
  // cpppatterns.com/patterns/virtual-constructor.html
  // some people call it "covariant return type"
  // Throws error from this class if not implemented
  /**
   * Return a copy of the op.
   * This method must be implemented. The compiler throws an error if this
   * method is not implemented.
   */
  virtual std::unique_ptr<Op> clone() const = 0;

  template <typename T> bool isConvertibleTo() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }

  /**
   * Check if this is a LossOp op, for example NllOp or L1Op.
   * \note The op SumOp which adds the losses together is not
   * a LossOp.
   * \returns `true` if this is a LossOp op, `false` otherwise.
   */
  virtual bool isLossOp() const;

  /**
   * Check if this is an IpuCopyOp op.
   * \returns `true` if this is an IpuCopyOp op, `false` otherwise.
   */
  virtual bool isIpuCopyOp() const;

  /**
   * Check if this copies only optimizer tensors from one IPU to another.
   * \returns `true` if this op copies only optimizer tensors from one IPU to
   *       another, `false` otherwise.
   */
  virtual bool copiesOptimizerTensors() const;

  /// Check if op is part of the optimizer.
  virtual bool isOptimizerOp() const;

  /// Check if op is a part of gradient clipping.
  bool isGradientClippingOp() const;

  // The random seed tensor used to set the IPU's random number generators is
  // created in the IR, and connected to the ops that require it.
  /**
   * Check if the op requires a random seed.
   * This is set to `false `by default and should be overridden and set to
   * `true` if an IPU random seed tensor is required by the op. If, so it will
   * be connected to inTensor(getSeedInIndex()) by the IR process.
   * \returns `true` if the op requires a random seed, `false` otherwise.
   */
  virtual bool requiresRandomSeed() const;

  /**
   * Check if the op requires a random seed.
   * This is set to false by default and should be overridden and set to true if
   * an IPU random seed tensor is required by the op. If, so it will be
   * connected to inTensor(getSeedInIndex()) by the IR process.
   * \returns `true` if the op requires a random seed, `false` otherwise.
   */
  virtual InIndex getSeedInIndex() const;

  /**
   * Check if the op has an input at the input index.
   * \returns `true` if the op has an input at the input index, otherwise
   * `false`.
   */
  bool hasInput(InIndex index) const;

  /**
   * Check if the op has an output at the output index.
   * \returns `true` if the op has an output at the output index, otherwise
   *       `false`.
   */
  bool hasOutput(OutIndex index) const;

  // helper functions, access fields of input and output

  /**
   * Get the input tensor at the input index.
   * \param index The input index.
   * \returns The tensor at the input index.
   */
  Tensor *inTensor(InIndex index);

  /**
   * Get the input tensor at the input index.
   * \param index The input index.
   * \returns The tensor at the input index.
   */
  const Tensor *inTensor(InIndex index) const;

  /**
   * Get the output tensor at the output index.
   * \param index The output index.
   * \returns The tensor at the output index.
   */
  Tensor *outTensor(OutIndex index);

  /**
   * Get the output tensor at the output index.
   * \param index The output index.
   * \returns The tensor at the output index.
   */
  const Tensor *outTensor(OutIndex index) const;

  /**
   * Get the ID of the input tensor at the input index.
   * \param index The input index.
   * \returns The tensor ID of the tensor at the input index.
   */
  TensorId inId(InIndex index);

  /**
   * Get the ID of the input tensor at the input index.
   * \param index The input index.
   * \returns The tensor ID of the tensor at the input index.
   */
  const TensorId inId(InIndex index) const;

  /**
   * Get the ID of the output tensor at the output index.
   * \param index The output index.
   * \returns The tensor ID of the tensor at the output index.
   */
  TensorId outId(OutIndex index);

  /**
   * Get the ID of the output tensor at the output index.
   * \param index The output index.
   * \returns The tensor ID of the tensor at the output index.
   */
  const TensorId outId(OutIndex index) const;

  /**
   * Get the info of the input tensor at the input index.
   * \param index The input index.
   * \returns The tensor info of the tensor at the input index.
   */
  TensorInfo &inInfo(InIndex index);

  /**
   * Get the info of the input tensor at the input index.
   * \param index The input index.
   * \returns The tensor info of the tensor at the input index.
   */
  const TensorInfo &inInfo(InIndex index) const;

  /**
   * Get the info of the output tensor at the output index.
   * \param index The output index.
   * \returns The tensor info of the tensor at the output index.
   */
  TensorInfo &outInfo(OutIndex index);

  /**
   * Get the info of the output tensor at the output index.
   * \param index The output index.
   * \returns The tensor info of the tensor at the output index.
   */
  const TensorInfo &outInfo(OutIndex index) const;

  /**
   * Get the shape info of the input tensor at the input index.
   * \param index The input index.
   * \returns The shape info of the tensor at the input index.
   */
  const Shape &inShape(InIndex index) const;

  /**
   * Get the shape info of the output tensor at the output index.
   * \param index The output index.
   * \returns The shape info of the tensor at the output index.
   */
  const Shape &outShape(OutIndex index) const;

  /**
   * Get the number of input tensors of this op.
   * \returns The number of input tensors this op has.
   */
  size_t inTensorCount() const;

  /**
   * Get the number of output tensors of this op.
   * \returns The number of output tensors this op has.
   */
  size_t outTensorCount() const;

  /**
   * Get the rank of the input tensor at the input index.
   * \param index The input index.
   * \returns The rank of the tensor at the input index.
   */
  Rank inRank(InIndex index) const;

  /**
   * Get the rank of the output tensor at the output index.
   * \param index The output index.
   * \returns The rank of the tensor at the output index.
   */
  Rank outRank(OutIndex index) const;

  /**
   * Get the input index of the tensor.
   * \param Tensor The input tensor.
   * \returns The input index of the tensor in the op.
   */
  InIndex inIndex(Tensor *) const;

  /**
   * Get the output index of the tensor.
   * \param Tensor The output tensor.
   * \returns The output index of the tensor in the op.
   */
  OutIndex outIndex(Tensor *) const;

  /**
   * Append attributes when serialising the op to a stream.
   * This is used for debugging and also to generate the PopART IR hash.
   * This hash is used to determine whether a Poplar cache can be reused so it
   * is important that op attributes which may alter the Poplar compilation are
   * appended to this stream. If this method is overridden, then it must also
   * call the base class method.
   * \param OpSerialiserBase The stream to which the attributes should be
   *       appended.
   */
  virtual void appendAttributes(OpSerialiserBase &) const;

  /**
   * Append the op attributes that are relevant for outlining ops.
   * Ops should override this function if there are additional attributes.
   * Two ops with identical type and outline attributes can be outlined and are
   * supposed to be functionally equivalent.
   * \param OpSerialiserBase The stream to which the attributes should be
   *       appended.
   */
  virtual void appendOutlineAttributes(OpSerialiserBase &) const;

  /**
   * Append additional attributes to the stream.
   * This method should be overridden if the derived class has additional
   * attributes.
   * \param OpSerialiserBase The stream to which the attributes should be
   *       appended.
   */
  virtual void appendMore(OpSerialiserBase &) const;

  /**
   * Calculate the NumPy broadcast shape for two shapes.
   * This will throw an error if the broadcast is not aligned. The error will
   * have operator context.
   * Note: If the replicated tensor sharding meta-shape is required, use
   * prettyNpOut with TensorInfo instead.
   * \param s0 The first shape.
   * \param s1 The second shape.
   * \return The NumPy-like broadcasted output shape.
   */
  Shape prettyNpOut(const Shape &s0, const Shape &s1) const;

  /**
   * Calculate the NumPy broadcast shape for two shapes.
   * This will throw an error if the broadcast is not aligned. The error will
   * have operator context.
   * \param i0 The info for the first tensor containing shape and meta-shape.
   * \param i1 The info for the second tensor containing shape and meta-shape.
   * \param checkDataType Check that the data types are identical.
   *      If `true`, check that the data types are identical and throw an
   *      error if they are not. If `false`, do not check that data types are
   *      identical.
   * \return The NumPy-like broadcast output info containing the correct
   *      shape and meta-shape. The data type is taken from \p i0.
   */
  TensorInfo prettyNpOut(const TensorInfo &i0,
                         const TensorInfo &i1,
                         bool checkDataType = true) const;

  /**
   * Get all graphs that this op may call during its execution.
   * \return A vector of all graphs that this op may call during its execution.
   */
  virtual std::vector<const Graph *> getCalledGraphs() const;

  /**
   * Get the IDs of all graphs that this op may call during its execution.
   * \return A vector of IDs of all graphs that this op may call during its
   *       execution.
   */
  std::vector<GraphId> getCalledGraphIds() const;

  /**
   * Get the index in the op where the graph is called.
   * \param id The ID of the called graph.
   * \return The index at which the graph is called.
   */
  SubgraphIndex getCalledGraphIndex(const GraphId &id) const;

  // For each subgraphIndex [0,getCalledGraphs().size()) and each valid InIndex
  // for this op, return the associated subgraph's InIndex (or return -1
  // to indicate it is not used by this subgraph).
  /**
   * Get the input index for the subgraph corresponding to the op input index.
   * \param subgraphIndex The index of the subgraph from the set of subgraphs
   *       called by this op (returned by getCalledGraphs()).
   * \param inIndex The input index in the op.
   * \returns The input index in the subgraph that corresponds to the input
   *       index in the op, or -1 if the op input index is not used by the
   *       subgraph.
   */
  virtual InIndex opInToSubgraphInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex) const;

  // For each subgraphIndex [0,getCalledGraphs().size()) and each valid
  // subgraph's input index, return the associated op's input index (or return
  // -1 to indicate there isn't one).
  /**
   * Get the input index for the op corresponding to the subgraph input index.
   * \param subgraphIndex The index of the subgraph from the set of subgraphs
   *       called by this op (returned by getCalledGraphs()).
   * \param inIndex The input index in the subgraph.
   * \returns The input index in the op that corresponds to the input
   *       index in the subgraph, or -1 if the subgraph input index is not used
   *       by the op.
   */
  virtual InIndex subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex) const;

  // For each subgraphIndex [0,getCalledGraphs().size()) and each valid OutIndex
  // for this op, return the associated subgraph's OutIndex (or return -1
  // to indicate it is not used by this subgraph).
  /**
   * Get the output index for the subgraph corresponding to the op output index.
   * \param subgraphIndex The index of the subgraph from the set of subgraphs
   *       called by this op (returned by getCalledGraphs()).
   * \param outIndex The output index in the op.
   * \returns The output index in the subgraph that corresponds to the output
   *       index in the op, or -1 if the op output index is not used by the
   *       subgraph.
   */
  virtual OutIndex opOutToSubgraphOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex) const;

  // For each subgraphIndex [0,getCalledGraphs().size()) and each valid
  // subgraph's output index, return the associated op's output index (or return
  // -1 to indicate there isn't one).
  /**
   * Get the output index for the op corresponding to the subgraph output index.
   * \param subgraphIndex The index of the subgraph from the set of subgraphs
   *       called by this op (returned by getCalledGraphs()).
   * \param outIndex The output index in the subgraph.
   * \returns The output index in the op that corresponds to the output
   *       index in the subgraph, or -1 if the subgraph output index is not used
   *       by the op.
   */
  virtual OutIndex subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex) const;

  /**
   * Get the the set of outputs to visit based on the input index (for graph
   * traversal).
   * \param in The input index used to determine the set of outputs to visit.
   * \returns The set of outputs to visit based on the input index.
   */
  virtual std::set<OutIndex> opInToOpOutIndex(InIndex in) const;

  /**
   * Get the the set of inputs to visit based on the output index (for graph
   * traversal).
   * \param out The output index used to determine the set of inputs to visit.
   * \return The set of inputs to visit based on the output index.
   */
  virtual std::set<InIndex> opOutToOpInIndex(OutIndex out) const;

public:
  /// The functionality required for sub-graph matching.
  using SubgraphInSig =
      std::tuple<Op *, fwtools::subgraph::OutIndex, std::string>;

  /**
   * Get a string that represents the equivalence class that this op belongs to.
   * This is used by, for example transforms, to determine if two ops are the
   * same. If and only if two ops return the same equivalence ID then those ops
   * can be considered of the same equivalence class.
   *
   * \param externalAttrs Additional attributes by which to distinguish this op.
   *    The value types must be one of: float, double, int, int64_t, uint32_t,
   *    uint64_t, std::string, std::vector<float>, std::vector<double>,
   *    std::vector<int64_t>, popart::Scope, bool, nonstd::optional<int64_t>,
   *    nonstd::optional<float>, nonstd::optional<double> or
   *    std::map<TensorId, uint64_t>. We use this to add, for example
   *    replica-equalness properties to the equivalence ID, which is a property
   *    that is calculated on-the-fly as opposed to stored in the op.
   * \return The equivalence ID.
   */
  std::string getSubgraphEquivId(
      const std::map<std::string, popart::any> &externalAttrs = {}) const;

  /**
   * Get all the producer ops of the tensors consumed at the input index.
   * \returns A map of producer ops for the tensors consumed at the input index.
   */
  std::map<fwtools::subgraph::InIndex, SubgraphInSig> getSubgraphInputs() const;

  /**
   * Get all the consumer ops of the tensors produced at the output index.
   * \returns A map of consumer ops for the tensors produced at the output
   * index.
   */
  std::map<fwtools::subgraph::OutIndex, std::set<Op *>>
  getSubgraphOutputs() const;

  // default high value here means that sub-graphs
  // of single ops are cached by default
  /**
   * Get the subgraph value.
   * This is used by outlining algorithm to determine whether or not to outline
   * ops. There are high bounding values retrieved by
   * getHighSubgraphValue() (for expensive ops such as Conv) or low bounding
   * values retrieved by getLowSubgraphValue() (for inexpensive ops such as
   * Relu).
   * \returns The subgraph value. Default: 0.
   */
  virtual float getSubgraphValue() const = 0;

  /// Return the high subgraph value.
  // for example, conv has this value in getSubgraphValue(),
  float getHighSubgraphValue() const { return 1000.0f; }

  /// Return the low subgraph value.
  // and relu has this value.
  float getLowSubgraphValue() const { return 0.1f; }

  /// Get approximate cost of activations between forward and backward graphs.
  virtual float calcAutoVirtualGraphCost(std::set<int> &inputs_seen);

  /**
   * Check if op can be outlined.
   * If this method returns `false`, it will mean that any possible subgraph
   * that this op is part of will not be cached.
   * \return `true` if the op can be outlined, `false` otherwise.
   *      Default: `true`.
   */
  virtual bool isOutlineable() const;

  /**
   * Check if the op has any effect that is not captured by the (modification
   * of) input or output tensors, such as modifying the state of the IPU or host
   * system.
   * \return `true` if the op has side effects, `false` otherwise.
   *      Default=`false`.
   */
  virtual bool hasSideEffect() const;

  /**
   * Check if the op can be recomputed.
   * To recompute an op means to clone it to produce the same output.
   * The function checks the safeness of recompute in the context of explicit
   * recompute. It may still be unsafe for implicit recompute.
   * \return `true` if the op can be recomputed, `false` otherwise.
   * Default: hasSideEffect().
   */
  virtual bool canRecompute() const;

  /**
   * Check if any input indices are unmodifiable or alias an unmodifiable
   * tensor.
   *
   * \returns `true` if any connected variable tensor for all input indices has
   *      a non-empty alias chain and is unmodifiable, `false` otherwise.
   */
  bool inputsUnmodifiable() const;

  /**
   * Check if op consumes the outputs of the graph.
   *
   * \returns `true` if op consumes graph outputs, `false` otherwise.
   */
  bool consumesGraphOutput() const;

  /**
   * Check if op produces the outputs of the graph.
   *
   * \returns `true` if op produces graph outputs, `false` otherwise.
   */
  bool producesGraphOutput() const;

  /**
   * Check if the input index is unmodifiable or aliases an unmodifiable tensor.
   *
   * \param in The input index to check.
   * \returns `true` if any connected variable tensor has a non-empty alias
   *      chain and is unmodifiable, `false` otherwise.
   */
  bool inputUnmodifiable(InIndex in) const;

  /**
   * Check if output is modified by any consumer.
   *
   * \param out The output index to check.
   * \returns `true` if any consumer of any aliased tensor downstream modifies
   *              a non-empty region, `false` otherwise.
   */
  bool hasAliasedModifiers(OutIndex out) const;

  // Helper functions for probing graph structure.
  /**
   * Check if the graph is a parent of the op.
   * A graph is a parent of an op if and only if the op is a child of the graph.
   * \param 1 The op that is being checked.
   * \returns `true` if the graph is a parent graph, `false` otherwise.
   */
  bool isParentOf(const Op *) const;
  /**
   * Check if the graph is a child graph.
   * A graph is a direct child of an op if the graph consumes any of the tensors
   * the op produces.
   * \param 1 The op that is being checked.
   * \returns `true` if the graph is a child graph, `false` otherwise.
   */
  bool isChildOf(const Op *) const;

  // TODO: T16743: extend support for other dimensions than the batch
  /**
   * Check if the operation can be sharded into multiple operations.
   * \returns `true` if the operation can be sharded, `false` otherwise.
   */
  virtual bool canShard() const;

  /**
   * Get the reduction type to apply after sharding, if the output shape does
   * not change.
   * \param index The output index at which to determine the reduction type.
   * \returns The reduction type.
   */
  virtual ReductionType getShardReductionType(OutIndex index) const;

  /**
   * Get the scale factor to apply after sharding, if required.
   * \param shardedOp The sharded op.
   * \param index The output index at which to determine the scale factor.
   * \returns The scale factor. Default:1.0.
   */
  virtual float getShardRescaleFactor(Op *const shardedOp,
                                      OutIndex index) const {
    return 1.0f;
  }

  // TODO: T16743: extend support for other dimensions than the batch
  /**
   * Shard an operation into multiple operations according to the new,
   * already sharded input tensors.
   * \param inputs The sharded input tensors.
   * \returns The sharded output tensors.
   */
  std::map<TensorId, std::vector<TensorId>>
  shard(const std::map<TensorId, std::vector<TensorId>> &inputs);

  /**
   * Create an output sharding plan from sharding an op.
   * The sharding plan also contains the individual input/output shards of an
   * operation. When sharding an operation, the new plan is updated with the
   * resulting sharded tensors.
   * \param plan The input sharding.
   * \returns The plan after sharding the operation containing the resulting
   * sharded tensors.
   */
  ShardingPlan shard(const ShardingPlan plan);

  /**
   * Configure a sharded op.
   * \param shardedOp The sharded op to be configured.
   * \param settings_ The settings to apply to the sharded op.
   */
  virtual void configureShardedOp(Op *const shardedOp,
                                  const Settings *const settings_) const;

  /// Return which inputs and outputs are replicated tensor sharding pairs.
  virtual ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const;

  /**
   * Configure the op for replicated tensor sharding at specific indices.
   * \param indices The indices at which to configure the op for replicated
   *      tensor sharding.
   * \param shardingDomain The type and size of the replica group specified by a
   *      CommGroup object.
   */
  virtual void
  configureForReplicatedTensorSharding(ReplicatedTensorShardingIndices indices,
                                       CommGroup shardingDomain);

  // T41400 is to remove all other versions of `transferBaseProperties` and
  // replace with this.
  /**
   * Transfer the base properties from another op to this op.
   * \param to The op to transfer the base properties from.
   */
  void transferBaseProperties(Op *to);

  /**
   * Get the producer op of the input tensor at the input index.
   *
   * \param inIndex The index at which the input tensor is produced.
   * \returns The op which produces the input tensor at the input index.
   */
  Op *getPrecedingOp(InIndex inIndex);

  /**
   * Get the op that consumes an output tensor at an output index.
   *
   * This will throw an error if there is more than one consumer op.
   * \param outIndex The index at which the output tensor is consumed.
   * \returns The op which consumes the output tensor at the output index.
   */
  Op *getFollowingOp(OutIndex outIndex = 0);

  /**
   * Get all ops that consume an output tensor at an output index.
   *
   * \param outIndex The index at which the output tensor is consumed.
   * \returns A vector of ops which consume the output tensor at the output
   *      index.
   */
  std::vector<Op *> getFollowingOps(OutIndex outIndex = 0);

  /**
   * Get the producer op of the input tensor at the input index.
   *
   * This will throw an error if the producer op cannot be converted to type
   * `T`. \param inIndex The index at which the input tensor is produced.
   * \returns The op, converted to type `T`, which produces the input
   *      tensor at the input index.
   */
  template <typename T> T *getPrecedingOp(InIndex inIndex) {
    auto x = getPrecedingOp(inIndex);
    if (T *result = dynamic_cast<T *>(x)) {
      return result;
    } else {
      throw internal_error(
          "Preceding op {} is not convertible to requested type.", debugName());
    }
  }

  /**
   * Get the op that consumes an output tensor at an output index.
   *
   * This will throw an error if there is more than one consumer op, or if the
   * consumer op cannot be converted to type `T`.
   * \param outIndex The index at which the output tensor is consumed.
   * \returns The op, converted to type `T`, which consumes the output
   *      tensor at the output index.
   */
  template <typename T> T *getFollowingOp(OutIndex outIndex = 0) {
    Op *x = getFollowingOp(outIndex);
    if (T *result = dynamic_cast<T *>(x)) {
      return result;
    } else {
      throw internal_error(
          "Following op {} is not convertible to requested type.", debugName());
    }
  }

  /**
   * Get all ops that consume an output tensor at an output index.
   *
   * This will throw an error if not all of the consumer ops can be converted
   * to type `T`.
   * \param outIndex The index at which the output tensor is consumed.
   * \returns A vector of ops, converted to type `T`, which consume the output
   *      tensor at the output index.
   */
  template <typename T>
  std::vector<T *> getFollowingOps(OutIndex outIndex = 0) {
    auto xs = getFollowingOps(outIndex);
    std::vector<T *> result;
    for (auto x : xs) {
      if (T *t = dynamic_cast<T *>(x)) {
        result.push_back(t);
      } else {
        throw internal_error("Not all ops following {} at out index {} are "
                             "convertible to requested type.",
                             debugName(),
                             outIndex);
      }
    }
    return result;
  }

  /**
   * Check if the op is of the class IpuCopyOp that copies between pipeline
   * stages.
   *
   * \return `true` if op is of the class IpuCopyOp and copies between pipeline
   * stages, `false` otherwise.
   */
  bool isPipelineIpuCopyOp() const;

protected:
  // Attempt to get the data of an input tensor. This method will throw an
  // exception if it could not access the data.
  void getInTensorData(TensorId tensorId,
                       std::vector<int64_t> &data,
                       std::vector<DataType> dataTypes = {DataType::INT64,
                                                          DataType::INT32});

private:
  std::pair<ShardingPlan, ShardingPlan>
  adjustShardPlans(const ShardingPlan inputPlan);
  ShardingPlan unrollShard(const ShardingPlan adjustedInputPlan,
                           ShardingPlan outputPlan);
  ShardingPlan loopShard(const ShardingPlan adjustedInputPlan,
                         ShardingPlan outputPlan);
};

std::ostream &operator<<(std::ostream &, const GradInOutMapper &);
std::ostream &operator<<(std::ostream &, const GradOpInType &);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_HPP_
