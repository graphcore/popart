// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OP_HPP
#define GUARD_NEURALNET_OP_HPP

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>
#include <popart/attributes.hpp>
#include <popart/basicoptionals.hpp>
#include <popart/bwdgraphinfo.hpp>
#include <popart/debugcontext.hpp>
#include <popart/names.hpp>
#include <popart/opdebuginfo.hpp>
#include <popart/opidentifier.hpp>
#include <popart/region.hpp>
#include <popart/scope.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensorlocation.hpp>
#include <popart/util.hpp>
#include <popart/vertex.hpp>

#include <popart/subgraph/subgraphnames.hpp>

namespace poprithms {
namespace memory {
namespace inplace {
class Proposal;
} // namespace inplace
} // namespace memory
} // namespace poprithms

namespace popart {

struct PoprithmsAliaser;

enum class RecomputeType { Undefined = 0, Checkpoint, Recompute, Recomputed };

enum class ExecutionContext {
  Normal = 0,
  AccumulateOuterFragment,
  WeightsFromHostFragment,
  WeightsToHostFragment,
  OptimizerFromHostFragment,
  Subgraph
};

/// Defines the type of reduction used when weight updates of a batch are
/// computed in one go and are reduced over the gradients of the whole
/// minibatch.
enum class ReductionType {
  /// Sum the output of the loss values and do not scale the gradient.
  Sum = 0,
  /// Take the mean of the loss values and divide the gradient by the number
  /// of samples.
  Mean,
  /// Leave the loss values as they are and do not scale the gradient.
  NoReduction,
  /// The number of ReductionType values.
  N
};

std::ostream &operator<<(std::ostream &, const RecomputeType &);
std::ostream &operator<<(std::ostream &, const ExecutionContext &);
std::ostream &operator<<(std::ostream &, const ReductionType &);

class OpSerialiserBase;

class ShardingPlan;

/// The relationship between the input tensor of a grad-op and the
/// corresponding non-grad-op.
// design note: it's not possible for an input to a
// grad-op to NOT be directly related to
// the corresponding non-grad-op.
enum class GradOpInType { In = 0, Out, GradOut };

class GradInOutMapper {
public:
  GradInOutMapper(InIndex iGrad_, int iNonGrad_, GradOpInType);
  // input index to a grad-op
  InIndex iGrad;
  // "input/output/gradient-of-output" index to
  // corresponding non-grad op,
  int iNonGrad;
  // where "input/output/gradient-of-output" above is
  GradOpInType type;

  bool operator==(const GradInOutMapper &rhs) const;
};

class Op : public Vertex {
public:
  // We use pointers to TensorIndexMaps for PIMPL reasons.
  // Note that we cannot initialise these with {nullptr} on gcc.
  // They are initialised in the Op constuctors
  // The consumed Tensors
  std::unique_ptr<TensorIndexMap> input;
  // The produced Tensors
  std::unique_ptr<TensorIndexMap> output;

  // The unique identifier of the Op (will always be set in Op::Op)
  OpId id{-1};

  // The operation type, domain & version
  //   A given operator is identified by a three-tuple: (domain, op_type, and
  //   op_version). This is written as domain.op_type:op_version in prose (e.g.,
  //   com.acme.FastConv:3). Nodes in graphs always refer to operators by their
  //   three-part identifier.
  OperatorIdentifier opid;

  bool pruneable = true;

  struct Settings {

    Settings(Graph &graph_, const std::string &name_)
        : graph(graph_), name(name_) {}
    Settings(Graph &graph_, const std::string &name_, const Scope &scope_)
        : graph(graph_), name(name_), scope(scope_) {}
    virtual ~Settings()        = default;
    Settings(const Settings &) = default;

    Settings copy(const std::string &new_name) {
      Settings s = *this;
      s.name     = new_name;
      return s;
    }

    std::reference_wrapper<Graph> graph;

    std::string name = "";

    Scope scope;
    RecomputeType recomputeType = RecomputeType::Undefined;
    OptionalTensorLocation tensorLocation;

    // optional inplace priorities, to take precedence over the default
    // priorities. A negative priority gurarantees no inplacing
    // This should really be a map with "OperatorIdentifier" keys, see T6783
    std::vector<std::tuple<std::string, float>> inplacePriorityVeto;

    // A set of patterns which should not be applied to this op.
    std::unordered_set<std::string> excludePatterns;

    // The virtual graph this op has been assigned to if set
    OptionalVGraphId vgraphId;

    OptionalPipelineStage pipelineStage;

    // The execution phase this op has been assigned to if set
    OptionalExecutionPhase executionPhase;

    OptionalBatchSerializedPhase batchSerializedPhase;

    // If the OP should be placed on I/O tiles instead of regular tiles
    TileSet tileSet{TileSet::Compute};

    // If the OP needs to run in a special fragment,
    // such as gradient accumulation
    ExecutionContext executionContext{ExecutionContext::Normal};

    // Tensor layout mapping should be inferred "to" tensor <- "from" tensor
    std::map<InIndex, InIndex> inferTensorMappingToFrom;

    // all Ops will be topologically sorted "as close to" the order of
    // priority (highest to lowest) while still resulting in a valid
    // topological ordering.
    // default : 0.0
    double schedulePriority{0.0};

    // Extra attributes to differentiate ops for outlining
    // Ops with different outline attributes are not outlined together
    std::map<std::string, std::string> extraOutlineAttributes;

    // The debug info id of the parent debug info for this op.
    uint64_t debugInfoId{0};

    // To flag an Op as being part of the optimizer
    bool optimizerOp{false};

    // This method will append the optional attributes (vgraphId, etc)
    // depending on whether the attribute has been
    // set in the onnx model.
    virtual void setFromAttributes(const Attributes &attributes);

    Ir &getIr() const;
  };

  Settings settings;

  // The Op debug information
  OpDebugInfo debugInfo;

  Settings &getSettings() { return settings; }
  const Settings &getSettings() const { return settings; }

  // Return suitable settings for an Op inserted before the input to an existing
  // Op
  virtual Settings getInSettings(InIndex) const;

  // Return suitable settings for an Op inserted after the output to an existing
  // Op
  virtual Settings getOutSettings(OutIndex) const;

  // Adjust the settings to be suitable as input at InIndex
  Settings adjustInSettings(InIndex, Op::Settings) const;

  // Adjust the settings to be suitable as output at OutIndex
  Settings adjustOutSettings(InIndex, Op::Settings) const;

  const OptionalVGraphId getOptionalVGraphId() const;
  VGraphId getVirtualGraphId() const;
  virtual VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex, std::set<OpId> visited = {}) const;
  virtual VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex,
                                    std::set<OpId> visited = {}) const;
  void setVirtualGraphId(const OptionalVGraphId);
  bool hasVirtualGraphId() const;

  const OptionalExecutionPhase getOptionalExecutionPhase() const;
  virtual ExecutionPhase getExecutionPhase() const;
  void setExecutionPhase(const OptionalExecutionPhase);
  bool hasExecutionPhase() const;

  const OptionalBatchSerializedPhase getOptionalBatchSerializedPhase() const;
  virtual BatchSerializedPhase getBatchSerializedPhase() const;
  void setBatchSerializedPhase(const OptionalBatchSerializedPhase);
  bool hasBatchSerializedPhase() const;

  bool isExcludedFromPattern(const Pattern *) const;

  void setPipelineStage(OptionalPipelineStage);
  bool hasPipelineStage() const;
  PipelineStage getPipelineStage() const;
  OptionalPipelineStage getOptionalPipelineStage() const;

  virtual int getInBatchAxis(InIndex) const { return 0; }
  virtual int getOutBatchAxis(OutIndex) const { return 0; }

  // Inherit placement attributes:
  // - Pipeline stage
  // - Execution phase
  // - Virtual graph ID
  // - Batch serial phase
  void inheritPlacementAttributes(bool inheritSerializations);

  Ir &getIr();
  const Ir &getIr() const;

  Graph &getGraph() { return settings.graph.get(); }
  const Graph &getGraph() const { return settings.graph.get(); }

  const Scope &getScope() const { return settings.scope; }
  void setScope(const Scope &scope) { settings.scope = scope; }

  const std::string &getName() const { return settings.name; }
  void setName(const std::string &name) { settings.name = name; }

  const OpDebugInfo &getDebugInfo() const { return debugInfo; }

  virtual bool isNorm() const;
  bool isElementWiseUnary() const;

  // Methods used by patterns to determine if an op can be replaced by another
  // op

  // Return true if the op based on it's configuration can be replace by the
  // identity operations, else false.
  virtual bool canBeReplacedByIdentity() const;

public:
  Op(const OperatorIdentifier &_opid, const Op::Settings &settings);

  // Note: copy constructor does NOT copy input and output
  Op(const Op &);
  Op &operator=(const Op &) = delete;
  // A c++ aside: the rule-of-3 says that it's good
  // practise to have an explicit destructor,
  // given that there is an explict copy con.
  // But not really nec. as Vertex has a virtual
  // destructor.
  virtual ~Op();

  std::string str() const final;
  std::string debugName() const;

  // create an ActGrad (output) tensor
  // and wire it to this Op's output
  void createAndConnectOutTensor(OutIndex, TensorId);

  void append(std::stringstream &ss) const;

  void toJSON(std::stringstream &ss) const;

  // sum of the total memory of all output tensors
  // We might want a cycle counter too for more sophisticated recomputation
  int64_t memOfOutputs() const;

  virtual std::set<InIndex> optionalInputs() const { return {}; }

  // wire a tensor to input: updates input and
  // updates consumers of tensor with id TensorId
  void defaultConnectInTensor(InIndex, TensorId);

  virtual void connectInTensor(InIndex, TensorId);

  void connectOutTensor(OutIndex, TensorId);

  // Disconnect an input test from the op
  void disconnectInTensor(Tensor *tensor);
  virtual void disconnectInTensor(InIndex, Tensor *tensor);
  void disconnectInTensor(InIndex);

  // Disconnect an output tensor from the op
  void disconnectOutTensor(Tensor *tensor);

  // Disconnect all input tensors
  void disconnectAllInputs();

  // Disconnect all output tensors
  void disconnectAllOutputs();

  const std::string &name() const;

  // set shape and type parameters,
  // This function MUST set output
  // TensorInfos for all outputs
  virtual void setup();

  // Called once the IR 'prepare' is complete to finalize DebugInfo
  void finalizeDebugInfo();

  // if the op has called graphs (that is, `!getCalledSubgraphs().empty()`) then
  // this method will get called prior to `getGradOps` to provide the op with
  // the information it needs to call the grad version of the called subgraphs.
  virtual void
  setCalledSubgraphGradInfo(const FwdGraphToBwdGraphInfo &calledGraphsGradInfo);

  // return a vector of 1 or several gradient Ops: for
  // obtaining the gradient of the inputs of this Op.
  // If this Op is already a gradient Op, throws error
  // Why is this not constant? For one, nOps counter increments.
  virtual std::vector<std::unique_ptr<Op>> getGradOps();

  // What are the variants of this Op (if any) which can
  // modify / alias the inputs at the given indices?
  // This function doesn't check for anchor violations
  // or topological order violations. When there are several,
  // they should be returned in descending order of preference

  virtual std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const;

  virtual std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const;

  /**
   * For certain tasks which involve analysing how Tensors alias each other,
   * such as inplacing, a poprithms::memory::inplace::Graph, which corresponds
   * to this Op's Graph, is constructed. The poprithms Graph can then be queried
   * for aliasing information, and can have algorithms run on it.
   *
   * To construct the poprithms Graph, each PopART Op defines what its
   * poprithms equivalents Op are. This method inserts this Op's
   * poprithms::memory::inplace::Op equivalents into the poprithms Graph, which
   * is the container \a popAliaser.
   *
   * \sa PoprithmsAliaser
   * */
  virtual void growAliaser(PoprithmsAliaser &popAliaser) const;

  /**
   * Translate an inplacing proposal, which replaces this non-inplace Op with an
   * inplace Op of type #inplaceId, into a poprithms equivalent.
   *
   * \param proposal The poprithms Proposal to set
   *
   * \param aliaser Contains the mapping between this Op's (PopART) Graph and
   *                the poprithms Graph.
   *
   * \param inplaceId The OperatorIdentifier to translate to the poprithms
   *                  equivalent.
   *
   *
   * This method is defined as a void method which sets a value passed by
   * reference, as opposed to a getter method, so that no poprithms headers need
   * to be included in this file.
   * */
  virtual void setProposal(poprithms::memory::inplace::Proposal &proposal,
                           const PoprithmsAliaser &aliaser,
                           OperatorIdentifier) const;

protected:
  /**
   * This method is a possible implementation of the virtual method
   * growAliaser, which can be used by Ops which do not have more than 1
   * variant (that is, they have no "inplace" variants), and do no non-trivial
   * view-changing. Examples: LSTM, Conv, Matmul, SumReduce, etc.
   * */
  void growAliaserMulti(PoprithmsAliaser &) const;

  /**
   * Set the Proposal to open the poprithms::AliasGate which corresponds to this
   * Op, at input index 0. For information on AliasGates and inplacing
   * proposals, see the poprithms memory::inplace project.
   * */
  virtual void setProposalGate0(poprithms::memory::inplace::Proposal &proposal,
                                const PoprithmsAliaser &aliaser,
                                OperatorIdentifier opId) const;

public:
  // The input Region which this Op modifies (for inplace ops)
  virtual view::Regions modifies(InIndex) const;
  // The input Region which this Op uses
  virtual view::Regions uses(InIndex) const;
  // The input Region which the output will alias (for inplace and view-changing
  // ops)
  virtual view::Regions aliases(InIndex, OutIndex) const;
  // Map used regions of the input to/from the output (we assume the same for
  // modifies, aliases, uses)
  virtual view::RegMap fwdRegMap(InIndex, OutIndex) const;
  virtual view::RegMap bwdRegMap(InIndex, OutIndex) const;

  /**
   * \return True if there is an input which aliases an output.
   * */
  bool doesAlias() const;

  bool isOutplace() const { return !doesAlias(); }

  /**
   * \return True if the input at \p inIndex aliases the output at \p outIndex.
   * */
  bool doesAlias(InIndex inIndex, OutIndex outIndex) const;

  /** Is modifies(i) non-empty for any input index i?
   *
   * \returns     True if modifies(i) is non-empty for any i,
   *              false otherwise.
   */
  bool modifies() const;

  /** Check if an op modifies a tensor at a specific index in.
   *
   * \param in    Index to check.
   * \returns     True if it modifies the tensor,
   *              false otherwise.
   */
  bool modifiesIndex(InIndex in) const;

  /** Check if an op overwrites a tensor at a specific index in.
   *
   * \param t     Tensor to check.
   * \returns     True if it overwrites the tensor,
   *              false otherwise.
   */
  bool overwritesTensor(Tensor *t) const;

  /**
   * Is this op a view changing op? E.g. does it not modify it's input, and is
   * it an inplace op? Set at each op level. Examples: ReshapeInplaceOp,
   * IdentityInplace, TransposeInplaceOp
   *
   * \returns true If this is a view changing inplace op
   * \returns false Otherwise
   */
  virtual bool isInplaceViewChange() const { return false; };
  /**
   * Same as above for outplace (non-inplace) ops.
   * Examples: ReshapeOp, IdentityOp, TransposeOp
   *
   * \returns true If this is a view changing outplace op
   * \returns false Otherwise
   */
  virtual bool isOutplaceViewChange() const { return false; };

  // A grad-op outputs an edge-gradient tensor dT at gradOpOutIndex.
  // dT is the edge-gradient of a tensor T which was the input
  // to grad-op's non-grad partner. At what index was T the input
  // of non-grad-op? If not relevant (non-grad-ops) throw an error
  virtual int getNonGradInIndex(int gradOpOutIndex) const;

  // For grad-ops, matching input indices to
  // corresponding IN/OUT/GRADOUT indices of
  // corresponding non-grad-op.
  // throws an error if not appropriate (non-grad ops)
  virtual const std::vector<GradInOutMapper> &gradInputInfo() const;

  // return the full map corresponding to getNonGradInIndex.
  // throws an error if not appropriate (non-grad)
  virtual const std::map<int, int> &gradOutToNonGradIn() const;

  // return a copy of self, similar to
  // cpppatterns.com/patterns/virtual-constructor.html
  // some people call it "covariant return type"
  // Throws error from this class if not implemented
  virtual std::unique_ptr<Op> clone() const = 0;

  template <typename T> bool isConvertibleTo() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }

  // Is this Op a LossOp (nll, l1loss, etc)? Note:
  // the Sum op which adds the losses together is not
  // a LossOp
  virtual bool isLossOp() const;

  virtual bool isIpuCopyOp() const;

  // Returns true for Ops that copy only optimizer tensors
  // from one IPU to another
  virtual bool copiesOptimizerTensors() const;

  // Op that is part of the optimizer
  virtual bool isOptimizerOp() const;

  // The random seed tensor used to set the IPU's RNGs is created
  // in the IR, and connected to the Ops that require it
  virtual bool requiresRandomSeed() const;
  virtual InIndex getSeedInIndex() const;

  bool hasInput(InIndex index) const;
  bool hasOutput(OutIndex index) const;

  // helper functions, access fields of input and output
  Tensor *inTensor(InIndex index);
  const Tensor *inTensor(InIndex index) const;
  Tensor *outTensor(OutIndex index);
  const Tensor *outTensor(OutIndex index) const;

  TensorId inId(InIndex index);
  const TensorId inId(InIndex index) const;
  TensorId outId(OutIndex index);
  const TensorId outId(OutIndex index) const;

  TensorInfo &inInfo(InIndex index);
  const TensorInfo &inInfo(InIndex index) const;
  TensorInfo &outInfo(OutIndex index);
  const TensorInfo &outInfo(OutIndex index) const;

  const Shape &inShape(InIndex index) const;
  const Shape &outShape(OutIndex index) const;

  size_t inTensorCount() const;
  size_t outTensorCount() const;

  Rank inRank(InIndex index) const;
  Rank outRank(OutIndex index) const;

  OutIndex outIndex(Tensor *) const;

  // Virtual method to append the op attributes to the stream. This method
  // should be overridden if the derived class has additional attributes.
  virtual void appendAttributes(OpSerialiserBase &) const;

  // Virtual method to append the op attributes that are relevant for outlining
  // ops. Ops should override this function if there are additional attributes.
  // Two ops with identical type and outline attributes can be outlined and are
  // supposed to be functionally equivalent.
  virtual void appendOutlineAttributes(OpSerialiserBase &) const;

  // Calculate numpy broadcast shape for two shapes or generate an error if
  // the broadcast is not aligned. The error will have operator context.
  Shape prettyNpOut(const Shape &s0, const Shape &s1) const;
  TensorInfo prettyNpOut(const TensorInfo &i0, const TensorInfo &i1) const;

  // All graph that this op may call during its execution
  virtual std::vector<const Graph *> getCalledGraphs() const;
  std::vector<GraphId> getCalledGraphIds() const;

  // The index of the called graph
  SubgraphIndex getCalledGraphIndex(const GraphId &id) const;

  // For each subgraphIndex [0,getCalledGraphs().size()) and each valid InIndex
  // for this op, return the associated subgraph's InIndex (or return -1
  // to indicate it is not used by this subgraph).
  virtual InIndex opInToSubgraphInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex);
  // For each subgraphIndex [0,getCalledGraphs().size()) and each valid
  // subgraph's input index, return the associated op's input index (or return
  // -1 to indicate there isn't one).
  virtual InIndex subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                        InIndex inIndex);
  // For each subgraphIndex [0,getCalledGraphs().size()) and each valid OutIndex
  // for this op, return the associated subgraph's OutIndex (or return -1
  // to indicate it is not used by this subgraph).
  virtual OutIndex opOutToSubgraphOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex);
  // For each subgraphIndex [0,getCalledGraphs().size()) and each valid
  // subgraph's output index, return the associated op's output index (or return
  // -1 to indicate there isn't one).
  virtual OutIndex subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                           OutIndex outIndex);

protected:
  virtual void appendMore(OpSerialiserBase &) const;

public:
  // The functionality required for sub-graph matching
  using SubgraphInSig =
      std::tuple<Op *, fwtools::subgraph::OutIndex, std::string>;

  std::string getSubgraphEquivId() const;

  std::map<fwtools::subgraph::InIndex, SubgraphInSig> getSubgraphInputs() const;

  // all the consumers at a given output index
  std::map<fwtools::subgraph::OutIndex, std::set<Op *>>
  getSubgraphOutputs() const;

  // default high value here means that sub-graphs
  // of single Ops are cached by default
  virtual float getSubgraphValue() const = 0;

  // for example, conv has this value in getSubgraphValue(),
  constexpr float getHighSubgraphValue() const { return 1000.0f; }
  // and relu has this value.
  constexpr float getLowSubgraphValue() const { return 0.1f; }
  // Get approximation of cost of activations between fwd/bwd graph.
  virtual float calcAutoVirtualGraphCost(std::set<int> &inputs_seen);

  // Allow an op to exclude itself from caching. If this method returns false
  // it will mean that any possiable subgraph that this op is part of will
  // not be cached. The default is enabled (return true)
  virtual bool isOutlineable() const;

  virtual bool hasSideEffect() const;

  bool inputsUnmodifiable() const;

  bool consumesGraphOutput() const;
  bool producesGraphOutput() const;

  /** Check if input is unmodifiable or aliases an unmodifiable tensor.
   *
   * \param in    InIndex to check.
   * \returns     True if any connected variable tensor has a non-empty alias
   *              chain and is unmodifiable, false otherwise.
   */
  bool inputUnmodifiable(InIndex in) const;

  /** Check if output is modified by any consumer.
   *
   * \param out   OutIndex to check.
   * \returns     True if any consumer of any aliased tensor downstream modifies
   *              a non-empty region, false otherwise.
   */
  bool hasAliasedModifiers(OutIndex out) const;

  // Helper functions for probing graph structure.
  bool isParentOf(const Op *) const;
  bool isChildOf(const Op *) const;

  // Test if the operation can be sharded into multiple operations
  // TODO: T16743: extend support for other dimensions than the batch
  virtual bool canShard() const;

  // Get the reduction type to apply after sharding,
  // if the output shape does not change
  virtual ReductionType getShardReductionType(OutIndex index) const;

  // Get the scale factor to apply after sharding, if required
  virtual float getShardRescaleFactor(Op *const shardedOp,
                                      OutIndex index) const {
    return 1.0f;
  }

  // Shard operation into multiple operations according to the new,
  // already sharded input tensors. Returns the sharded output tensors.
  // TODO: T16743: extend support for other dimensions than the batch
  std::map<TensorId, std::vector<TensorId>>
  shard(const std::map<TensorId, std::vector<TensorId>> &inputs);

  ShardingPlan shard(const ShardingPlan plan);

  // Configure attributes/settings on sharded op
  virtual void configureShardedOp(Op *const shardedOp,
                                  const Settings *const settings_) const;

  // Return which inputs/outputs are replicated tensor sharding pairs
  virtual ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const;

  // Configure the operation for replicated tensor sharding at the specific
  // indices
  virtual void
  configureForReplicatedTensorSharding(ReplicatedTensorShardingIndices indices);

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

// A note on non-determinism. For maps with
// pointers as keys, iterating through them
// is non-deterministic with the default comparator.
/// To prevent non-determinism, POpCmp is used on any sets and maps that use
/// pointers to operators as a set/map key.
struct POpCmp {
  bool operator()(Op *const &a, Op *const &b) const { return a->id < b->id; }
};

struct POpIntCmp {
  bool operator()(std::pair<Op *, int> const &a,
                  std::pair<Op *, int> const &b) const {
    return std::pair<OpId, int>(a.first->id, a.second) <
           std::pair<OpId, int>(b.first->id, b.second);
  }
};

} // namespace popart

#endif
