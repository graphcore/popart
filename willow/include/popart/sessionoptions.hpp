// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_OPTION_FLAGS_HPP
#define GUARD_OPTION_FLAGS_HPP

#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <string>

#include <popart/op.hpp>
#include <popart/op/loss.hpp>
#include <popart/tensorlocation.hpp>
#include <popart/vendored/optional.hpp>

// Note that comments in this file have to adhere to doxygen formatting. See
// https://www.doxygen.nl/manual/.

namespace popart {

/**
 * A structure containing user configuration for automatic loss scaling
 * settings.
 *
 * **Note:** Automatic loss scaling is currently experimental and under
 * active development. We recommend that the user sets the loss scale
 * manually.
 */
struct AutomaticLossScalingSettings {
  AutomaticLossScalingSettings() = default;
  AutomaticLossScalingSettings(
      bool enabled_,
      const nonstd::optional<std::vector<TensorId>> &toTrackTensors_,
      float binEdgeLocation_               = 0.0625f,
      float thresholdUpperCountProportion_ = 1e-7,
      int updatePeriod_                    = 1);

  AutomaticLossScalingSettings &
  operator=(const AutomaticLossScalingSettings &rhs) = default;

  std::size_t hash() const;
  /// If true, keep track of the distribution of gradient tensor elements over
  /// the floating point range. Adjust the value loss scaling tensor
  /// accordingly, with the aim of preventing underflow or overflow.
  bool enabled = false;

  /// The location of the bin edge as a proportion of the absolute numerical
  /// range of the tracked gradient tensor elements, in the range [0, 1]. `0`
  /// represents the smallest representable value, and `1` the maximum. This is
  /// the single bin edge of the histogram that is an input to the loss scale
  /// updater algorithm.
  float binEdgeLocation = 0.0625f;

  /// The proportion of the elements in the upper bin above which the loss scale
  /// is increased, and below which the loss scale is decreased. Should be in
  /// the range [0, 1].
  float thresholdUpperCountProportion = 1e-7;

  /// An optional list of model tensor names, for which gradient statistics
  /// will be collected. If unset, the gradients of all tensors produced
  /// by a default operations (matmul, conv) will be used.
  nonstd::optional<std::vector<TensorId>> toTrackTensors;

  /// How often loss scale update factor should be updated with respect to
  /// optimizer steps.
  int updatePeriod = 1;
};

/**
 * Enum type to specify which ops to recompute in the backwards pass when doing
 * auto-recomputation.
 */
enum class RecomputationType {
  /// No ops are recomputed.
  None = 0,
  /// Algorithm to pick checkpoints to try and minimise max liveness.
  Standard,
  /// Only Norm ops (+ non-linearities, if following) are recomputed.
  NormOnly,
  /// Recompute all forward pipeline stages.
  Pipeline,
  /// Recompute all ops.
  RecomputeAll,
  /// The number of \c RecomputationTypes values.
  N
};

/**
 * Enum type used to specify which `VarUpdateOp` ops to merge.
 */
enum class MergeVarUpdateType {
  /// Do not merge VarUpdateOp ops.
  None = 0,
  /// Merge all VarUpdateOp ops into as few groups as possible.
  /// This is a good choice when memory is not a constraint.
  All,
  /// Merge into groups while attempting not to increase maximum
  /// variable liveness, and also not slice tensor variables so
  /// they will need to be processed by different VarUpdateOp ops.
  AutoLoose,
  /// Merge into groups, so that VarUpdateOp ops process tensors of
  /// exactly `mergeVarUpdateMemThreshold` in size.
  AutoTight,
  /// The number of \c MergeVarUpdateTypes values.
  N
};

/**
 * Enum type used to specify a virtual graph mode.
 */
enum class VirtualGraphMode {
  /// Virtual graphs are not enabled.
  Off = 0,
  /// User must set the `virtualGraph` attribute on all ops.
  Manual,
  /// Use `autoVirtualGraph` transform.
  Auto,
  /// Virtual graphs are tied to execution phases.
  ExecutionPhases,
  /// The number of \c VirtualGraphModes values.
  N
};

/**
 * Enum type used to specify a serialization format.
 */
enum class IrSerializationFormat {
  /// JavaScript Object Notation (JSON).
  JSON
};

/**
 * Enum type used to specify the data source for input tensors.
 */
enum class SyntheticDataMode {
  /// Use real data.
  Off = 0,
  /// Input tensors are initialised to all zeros.
  Zeros,
  /// Input tensors are initialised with distribution ~N(0,1).
  RandomNormal,
  /// The number of \c SyntheticDataMode values.
  N
};

/**
 * Enum type used to specify an instrumentation type.
 */
enum class Instrumentation {
  /// Outer loop instrumentation, graph over all IPUs.
  Outer = 0,
  /// Inner loop instrumentation, graph per IPU.
  Inner,
  /// The number of \c Instrumentation values.
  N
};

std::string toString(VirtualGraphMode);
std::ostream &operator<<(std::ostream &, VirtualGraphMode);

std::string toString(RecomputationType);
std::ostream &operator<<(std::ostream &, RecomputationType);

/**
 * A structure containing user configuration for cache/offloading settings.
 */
struct TensorLocationSettings {

  TensorLocationSettings() = default;
  TensorLocationSettings(TensorLocation location_,
                         int minElementsForOffChip_                  = 2,
                         int minElementsForReplicatedTensorSharding_ = 8192);

  TensorLocationSettings(TensorStorage storage_,
                         int minElementsForOffChip_                  = 2,
                         int minElementsForReplicatedTensorSharding_ = 8192);

  TensorLocationSettings &
  operator=(const TensorLocationSettings &rhs) = default;

  /// The default tensor location for this tensor type.
  TensorLocation location = TensorLocation();

  /// A minimum number of elements below which offloading won't be considered.
  int minElementsForOffChip = 2;

  /// A minimum number of elements below which replicated tensor sharding (RTS)
  /// won't be considered.
  int minElementsForReplicatedTensorSharding = 8192;
};

std::string toString(const TensorLocationSettings &);
std::ostream &operator<<(std::ostream &, const TensorLocationSettings &);

/**
 * Enum type that describes how to change the batch serialisation subgraph
 * schedule before outlining. \b NOTE: This setting is experimental and may
 * change.
 */
enum class BatchSerializationBatchSchedule {
  /// Don't encourage any particular scheduling for ops within batch subgraphs
  /// (leave it to the scheduler) but tell the scheduler to schedule subgraphs
  /// in sequence.
  Scheduler = 0,
  /// Encourage all ops within batch subgraphs to be scheduled identically and
  /// for each subgraph to be scheduled in sequence (good for outlineability).
  Isomorphic,
  /// Attempt to put the RemoteLoad for batch N+1 right after the
  /// compute phase of batch N.
  OverlapOnIo,
  /// Attempt to put the RemoteLoad for batch N+1 right before
  /// the compute phase of batch N.
  OverlapOnCompute,
  /// The number of \c BatchSerializationBatchSchedule values.
  N
};

/**
 * Enum type that describes when to apply the batch serialization.
 * \b NOTE: This setting is experimental and may change.
 */
enum class BatchSerializationTransformContext {
  /// Apply before growing the backward pass
  Fwd = 0,
  /// Apply after growing the backward pass
  Bwd,
  /// The number of \c BatchSerializationTransformContext values.
  N
};

/**
 * Enum type that describes how to apply the batch serialization.
 * \b NOTE: This setting is experimental and may change.
 */
enum class BatchSerializationMethod {
  /// Unroll the batch with dynamic slicing
  UnrollDynamic = 0,
  /// Unroll the batch with static slicing
  UnrollStatic,
  /// Loop over the batch dimension
  Loop,
  /// The number of \c BatchSerializationMethod values.
  N
};

/**
 * A structure containing batch serialization settings.
 */
struct BatchSerializationSettings {
  BatchSerializationSettings() = default;
  BatchSerializationSettings(
      int factor_,
      bool concatOnVirtualGraphChange_,
      bool concatOnExecutionPhaseChange_,
      bool concatOnPipelineStageChange_,
      BatchSerializationTransformContext transformContext_ =
          BatchSerializationTransformContext::Fwd,
      BatchSerializationMethod method_ =
          BatchSerializationMethod::UnrollDynamic,
      BatchSerializationBatchSchedule batchSchedule_ =
          BatchSerializationBatchSchedule::Isomorphic);

  BatchSerializationSettings &
  operator=(const BatchSerializationSettings &rhs) = default;
  /// The number of compute batches to split operations into.
  int factor = 0;
  /// Break batch serialization chains when the virtual graph
  /// changes (by concatenating the compute batches to the local batch).
  bool concatOnVirtualGraphChange = true;
  /// Break batch serialization chains when the execution phase
  /// changes (by concatenating the compute batches to the local batch).
  bool concatOnExecutionPhaseChange = true;
  /// Break batch serialization chains when the pipeline stage
  /// changes (by concatenating the compute batches to the local batch).
  bool concatOnPipelineStageChange = true;
  /// Experimental value to control when batch serialization is applied.
  BatchSerializationTransformContext transformContext =
      BatchSerializationTransformContext::Fwd;
  /// Experimental value to control how batch serialization is applied.
  BatchSerializationMethod method = BatchSerializationMethod::UnrollDynamic;
  /// Experimental value that changes how operations are scheduled.
  BatchSerializationBatchSchedule batchSchedule =
      BatchSerializationBatchSchedule::Isomorphic;
};

/**
 * Enum type to specify when to load tensors.
 */
enum class ExecutionPhaseIOSchedule {
  /// Preload tensors in previous phase for use in current phase.
  Preload = 0,
  /// Load tensors just before they are required.
  OnDemand,
  /// The number of \c ExecutionPhaseIOSchedule values.
  N
};

/**
 * Enum type to specify the order of processing optimizer operations for
 * different weights of the same execution phase.
 *
 * The steps for phased execution consists of:
 *
 * - Copy to IO tiles if necessary (1)
 * - Run collective operations if necessary (2)
 * - Load optimizer state (3)
 * - Update optimizer state (4)
 * - Apply optimizer (5)
 * - Store updated tensor if necessary (6)
 */
enum class ExecutionPhaseSchedule {
  /// Process above steps for one weight at a time (for example: 123456, 123456,
  /// 123456). The scheduler may interleave these steps.
  Interleaving = 0,
  /// Process above steps for all weights together, in a way that maximises
  /// overlap potential between compute and exchange
  /// (for example: 333, 111, 222, 444, 555, 666).
  Batch,
  /// Process above steps for all weights together, in a way that maximises
  /// overlap potential between compute and exchange, and maximise stream
  /// copy merges by keeping RemoteLoad/RemoteStore operations clustered
  /// (for example: 333, 111, 222, 444, 555, 666).
  BatchClusteredIO,
  /// The number of \c ExecutionPhaseSchedule values.
  N
};

/**
 * A structure containing ExecutionPhase settings.
 */
struct ExecutionPhaseSettings {
  ExecutionPhaseSettings() = default;
  ExecutionPhaseSettings(int phases_,
                         bool stages_,
                         ExecutionPhaseIOSchedule weightIOSchedule_,
                         ExecutionPhaseIOSchedule activationIOSchedule_,
                         ExecutionPhaseIOSchedule optimizerStateIOSchedule_,
                         ExecutionPhaseIOSchedule accumulatorIOSchedule_,
                         ExecutionPhaseSchedule schedule_)
      : phases{phases_}, stages{stages_}, weightIOSchedule{weightIOSchedule_},
        activationIOSchedule{activationIOSchedule_},
        optimizerStateIOSchedule{optimizerStateIOSchedule_},
        accumulatorIOSchedule{accumulatorIOSchedule_}, schedule{schedule_} {}

  ExecutionPhaseSettings &
  operator=(const ExecutionPhaseSettings &rhs) = default;

  /// Number of ExecutionPhases for the whole model
  int phases = 0;

  /// Number of overlapping stages
  /// 1: Parallel streaming memory, default for 1 IPU / replica
  /// 2: PingPong between 2 IPUs, default for >= 2 IPUs / replica
  int stages = 2;

  /// The execution phase IO schedule for weight tensors.
  ExecutionPhaseIOSchedule weightIOSchedule = ExecutionPhaseIOSchedule::Preload;
  /// The execution phase IO schedule for activation and gradient tensors.
  ExecutionPhaseIOSchedule activationIOSchedule =
      ExecutionPhaseIOSchedule::Preload;

  // TODO T28529: Add doxygen comments.
  ExecutionPhaseIOSchedule optimizerStateIOSchedule =
      ExecutionPhaseIOSchedule::OnDemand;
  ExecutionPhaseIOSchedule accumulatorIOSchedule =
      ExecutionPhaseIOSchedule::Preload;
  ExecutionPhaseSchedule schedule = ExecutionPhaseSchedule::Interleaving;
};

/**
 * Enum type that determines how the operations in the accumulate outer fragment
 * will be scheduled accross virtual graphs (only relevant to pipelined modes).
 */
enum class AccumulateOuterFragmentSchedule {
  /// Don't add additional constraints and let the scheduler work it out.
  Scheduler = 0,
  /// Add constraints that ensure ops are executed in virtual graph ID order.
  Serial,
  /// Try and parallelise ops with different virtual graph IDs as much as
  /// possible.
  OverlapCycleOptimized,
  /// Try and parallelise ops with different virtual graph IDs but avoid certain
  /// steps that are costly in terms of memory usage.
  OverlapMemoryOptimized
};

/**
 * A structure containing accumulate outer fragment settings.
 */
struct AccumulateOuterFragmentSettings {
  AccumulateOuterFragmentSettings() = default;
  AccumulateOuterFragmentSettings(
      AccumulateOuterFragmentSchedule schedule_,
      const std::vector<int> &excludedVirtualGraphs_)
      : schedule{schedule_}, excludedVirtualGraphs{excludedVirtualGraphs_} {}

  AccumulateOuterFragmentSettings &
  operator=(const AccumulateOuterFragmentSettings &rhs) = default;

  /// Tell PopART how you would like to schedule the accumulate outer fragment.
  /// This setting is experimental and may change.
  AccumulateOuterFragmentSchedule schedule =
      AccumulateOuterFragmentSchedule::Serial;
  /// A setting to explicitly tell PopART to avoid to try and parallelise the
  /// given virtual graph ids. This setting is experimental and may change.
  std::vector<int> excludedVirtualGraphs = {};
};

/**
 * Enum type that describes how copies for inputs and outputs for subgraphs
 * are lowered. Currently this only affects subgraphs associated with CallOps.
 */
enum class SubgraphCopyingStrategy {
  /// Copy all inputs before the start of the subgraph, copy all outputs after
  /// all ops in the subgraph. With this strategy subgraphs will always map
  /// to a single Poplar function.
  OnEnterAndExit = 0,
  /// Copy inputs just before they are consumed and copy outputs as soon as
  /// they are produced. With this strategy subgraphs may be lowered into
  /// multiple Poplar functions.
  JustInTime,
  /// The number of \c SubgraphCopyingStrategy values.
  N
};

/**
 * Enum type that specifies when to divide by a mean reduction factor, when
 * doing mean reduction over a sequence of tensors \f$t_1, t_2, ..., t_k\f$.
 */
enum class MeanReductionStrategy {
  /// Keep the reduction buffer as the mean of the tensors accumulated so far.
  /// If we have just processed \f$t_1, ..., t_f\f$,
  /// the current accumulator \f$s\f$ is the mean of these values, and
  /// the next accumulator update is
  /// \f$s = (f/(f+1)) * s + (1/(f+1)) * t_{f+1}\f$ to keep \f$s\f$ a running
  /// mean.
  ///
  /// This strategy guarantees \f$s \le \max(a_1, ..., a_k)\f$ throughout the
  /// accumulation, therefore it will not overflow, but it is generally slower
  /// than Post.
  Running = 0,
  /// Keep the accumulation factor as the running sum,
  /// and divide by \f$k\f$ once at the end of the accumulation.
  /// This strategy will generally be faster than Running,
  /// but is prone to overflow (especially when using `fp16`).
  Post,
  /// The number of \c MeanReductionStrategy values.
  N
};

/**
 * Type representing a strategy to ensure a backwards graph's inputs are
 * either inputs of the forward graph, outputs of the forward graph or
 * gradients of outputs of the forward graph. Strategies may expose tensors
 * that would otherwise have been internal to the forward graph as outputs of
 * said forward graph.
 **/
enum class AutodiffStitchStrategy {
  /// Recompute any backward graph inputs associated with non-gradient forward
  /// graph tensors that are neither inputs nor outputs in the forward graph.
  RecomputeMinimal = 0,
  /// Recompute any backward graph inputs associated with non-gradient forward
  /// graph tensors that are not inputs in the forward graph.
  RecomputeAllNonInputs,
  /// For backward graph inputs associated with non-gradient forward graph
  /// tensors that are neither inputs or outputs in the forward graph, add them
  /// as outputs to the forward graph.
  ///
  /// NOTE: This strategy is not guaranteed to work for all circumstances. In
  /// particular, it is unable to deal with subgraphs of IfOp. Using this
  /// setting may therefore result in subsequent exceptions in the autodiff
  /// transform and it is therefore inadvisable to use this as an `Autodiff`
  /// default.
  AddFwdOutputs,
  /// Like `AddFwdOutputs` except that those backward graph inputs that can't be
  /// stitched with `AddFwdOutputs` (that is, by adding outputs to the forward
  /// graph) are stitched using the `RecomputeMinimal` strategy instead. This
  /// means that this is a safe strategy to use as an `Autodiff` default.
  SafeAddFwdOutputs,
  /// Number of \c AutodiffStitchStrategy values.
  N
};

/**
 * Settings for the Autodiff transform.
 */
struct AutodiffSettings {
  AutodiffSettings() = default;
  AutodiffSettings(AutodiffStitchStrategy stitchStrategy_)
      : stitchStrategy{stitchStrategy_} {}

  AutodiffSettings &operator=(const AutodiffSettings &rhs) = default;

  /// The strategy PopART should use to ensure that all graph inputs of a
  /// backwards graph are available as either inputs or outputs of the forward
  /// graph or gradients of outputs of the forward graph.
  ///
  /// NOTE: This is an experimental option and may change.
  AutodiffStitchStrategy stitchStrategy =
      AutodiffStitchStrategy::RecomputeAllNonInputs;
};

// Struct for development-specific configuration intended to be used by
// PopART developers, as opposed to PopART users.
//
/// NOTE: These options are not subject to deprecation notices and may be
// changed or removed at any time.
struct DeveloperSettings {

  DeveloperSettings &operator=(const DeveloperSettings &rhs) = default;
  // The minimum percentage of the total time a scope must take in order
  // for it to be logged by the TimePartitionLogger.
  double timePartitionLoggerThresholdPercentage = 1.0f;
};

/**
 * A structure containing user configuration options for the Session class.
 */
struct SessionOptions {

  SessionOptions &operator=(const SessionOptions &rhs) = default;

  /// A directory for log traces to be written into.
  std::string logDir;

  /// When to write `.dot` files during Ir construction.
  std::set<std::string> dotChecks = {};

  /// The ops to write to the `.dot` file will be a continuous interval
  /// of the schedule, controlled by firstDotOp and finalDotOp. In particular,
  /// it will be [min(0, firstDotOp), max(N ops in Ir, finalDotOp)).
  int firstDotOp = 0;
  /// See #firstDotOp.
  int finalDotOp = 10000;

  /// Include the Op name in the `.dot` file (the Op type is always exported).
  bool dotOpNames = false;

  /// Export Poplar computation graph.
  bool exportPoplarComputationGraph = false;

  /// Export Poplar vertex graph.
  bool exportPoplarVertexGraph = false;

  /// When generating PDFs of IR graphs, create separate PDFs for each subgraph.
  bool separateCallOpPdfs = true;

  /// Identify and extract repeated parts of computational graph into subgraphs.
  bool enableOutlining = true;

  /// When `true` the cost of copying of cached sections should be included
  /// in the outlining cost model.
  bool enableOutliningCopyCostPruning = true;

  /// The incremental value that a sub-graph requires, relative to its nested
  /// sub-graphs (if any), to be eligible for outlining. A high threshold
  /// results in fewer sub-graphs being outlined, a negative value results in
  /// all being outlined. The gross value of a sub-graph is the sum of its
  /// constituent Ops' Op::getSubgraphValue() values. To disable outlining, it
  /// is better to set enableOutlining to false than to set this value to
  /// infinity. The default value of 1.0f results in all high value operations
  /// such as convolution being cached, but standalone low Value operations such
  /// as Relu will not be.
  float outlineThreshold = 1.0f;

  /// The penalty applied to outlining potential sub-graphs if the sub-graph
  /// to be created breaks up a sequence of operations that are more efficient
  /// (for example for overlapping compute and exchange) when outlined together.
  /// Default value is set to ~10 * Op::getHighSubgraphValue().
  float outlineSequenceBreakCost = 10000.0f;

  /// This setting determines how copies for inputs and outputs for subgraphs
  /// are lowered. By setting this value to JustInTime you may save memory at
  /// the cost of fragmenting subgraphs into multiple Poplar functions. This
  /// may be particularly useful when a number of weight updates are outlined
  /// in one subgraph, as it may prevent multiple weight tensors from being
  /// live at the same time inside the subgraph.
  SubgraphCopyingStrategy subgraphCopyingStrategy =
      SubgraphCopyingStrategy::OnEnterAndExit;

  /// Enable recomputation of operations in the graph in the backwards pass to
  /// reduce model size at the cost of computation cycles.
  RecomputationType autoRecomputation = RecomputationType::None;

  /// Enable merging of VarUpdates into groups of VarUpdates, by flattening
  /// and concatenating variable tensors and updating tensors.
  MergeVarUpdateType mergeVarUpdate = MergeVarUpdateType::None;

  /// The #MergeVarUpdateType::AutoLoose and #MergeVarUpdateType::AutoTight
  /// VarUpdateOp merging algorithms have a threshold on the total memory of
  /// variable tensors to merge for updating. Defined as total memory in bytes.
  int64_t mergeVarUpdateMemThreshold = 1000000;

  /// The #MergeVarUpdateType::AutoLoose VarUpdateOp merging algorithm has an
  /// absolute threshold defined by:
  ///
  /// \c min(#mergeVarUpdateMemThreshold, \c liveAtPeak - \c liveCurrently +
  /// #looseThresholdAtPeak)
  ///
  /// where:
  ///  * \c liveAtPeak is an estimate of the maximum live memory of the
  ///    computation; and
  ///  * \c liveCurrently is an estimate of the live memory where the
  ///    threshold is being used to determine whether to schedule or postpone a
  ///    VarUpdateOp.
  int64_t looseThresholdAtPeak = 8000;

  /// Before anchor tensors are streamed from device to host, they are not
  /// necessarily arranged in memory as required when they are to be copied
  /// from host stream to host. This can be done on the device or on the host.
  /// Done on host by default to save memory, but often at the expense of
  /// cycles, especially for larger anchor tensors.
  bool rearrangeAnchorsOnHost = true;

  /// Before stream tensors are streamed from host to device, they are not
  /// necessarily arranged in memory as required when they are to be copied
  /// from host stream to device. This can be done on the device or on the host.
  /// Done on device by default.
  bool rearrangeStreamsOnHost = false;

  /// By default, we will use prefetching for input data streams. Poplar will
  /// speculatively read data for a stream before is is required to allow the
  /// 'preparation' of the data to occur in parallel with compute.
  bool enablePrefetchDatastreams = true;

  /// When #enablePrefetchDatastreams is set this is the default buffering
  /// depth value used for streams that are not re-arranged on the host.
  /// This value can be overridden via #prefetchBufferingDepthMap.
  unsigned defaultPrefetchBufferingDepth = 1;

  /// When #enablePrefetchDatastreams is set this mapping can be used to set
  /// stream-specific buffering depths. This buffering depth could be envisaged
  /// as being the size of a circular buffer that feeds data to and from Poplar.
  /// A buffering depth greater than 1 may improve the performance due to
  /// increased parallelisation but comes at the cost of increasing the memory
  /// footprint. Streams for tensors that have no entry in this map default to a
  /// buffering depth of #defaultPrefetchBufferingDepth.
  std::map<TensorId, unsigned> prefetchBufferingDepthMap;

  /// By default, we use the stable softmax Poplar function. The input tensor
  /// to softmax, *x*, is preprocessed by subtracting max(*x*) from each element
  /// before computing the exponentials, ensuring numerical stability. If you
  /// are sure the inputs to your softmax operations are small enough to not
  /// cause overflow when computing the exponential, you can enable the
  /// non-stable version instead, to increase the speed.
  bool enableNonStableSoftmax = false;

  /// Enable replication of graphs.
  bool enableReplicatedGraphs = false;

  /// Enable gradient accumulation.
  bool enableGradientAccumulation = false;

  /// Specify how gradients are reduced when using gradient accumulation
  /// and graph replication.
  ReductionType accumulationAndReplicationReductionType = ReductionType::Sum;

  /// Specify when to divide by a mean reduction factor when
  /// accumulationAndReplicationReductionType is set to ReductionType::Mean.
  MeanReductionStrategy meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Post;

  /// If enableReplicatedGraphs is true, \c replicatedGraphCount will set the
  /// number of model replications. For example, if your model uses 1 IPU, a
  /// \c replicatedGraphCount of 2 will use 2 IPUs. If your model is
  /// pipelined across 4 IPUs, a \c replicatedGraphCount of 4 will use 16 IPUs
  /// total. Therefore, the number of IPUs you request must be a multiple of
  /// \c replicatedGraphCount. If the training is done across multiple instances
  /// then the \c replicatedGraphCount is the number of replicas for this
  /// instance.
  int64_t replicatedGraphCount = 1;

  /// Specify the number of micro-batches to accumulate before applying the
  /// varUpdate.
  int64_t accumulationFactor = 1;

  /// This option allows you to place ops on virtual graphs to achieve model
  /// parallelism - either manually using model annotations, or automatically.
  VirtualGraphMode virtualGraphMode = VirtualGraphMode::Off;

  /// Enable pipelining of virtual graphs
  bool enablePipelining = false;

  /// This options specifies whether to use real or synthetic data to initialize
  /// input tensors. Anything but the #SyntheticDataMode::Off value disables
  /// streaming to/from host.
  SyntheticDataMode syntheticDataMode = SyntheticDataMode::Off;

  /// Add instrumentation to your program to count the number of device cycles
  /// (of a single tile, on a single IPU) that your main program takes to
  /// execute. Expect this to have a small detrimental impact on performance.
  bool instrumentWithHardwareCycleCounter            = false;
  std::set<Instrumentation> hardwareInstrumentations = {Instrumentation::Outer};

  /// If true, the weight gradient tensors are not saved off the device
  /// when \c devicex.weightsFromHost() is called. Note: this option is
  /// overridden if #syntheticDataMode is not #SyntheticDataMode::Off.
  bool disableGradAccumulationTensorStreams = false;

  /// If false, the backend will build the Poplar graph but not compile it
  /// into an Engine.  In this case, no execution can be performed,
  /// and nothing can be transferred to the device. API calls which retrieve
  /// information from the graph building stage, such as tile mapping
  /// introspection, can still be used.
  bool compileEngine = true;

  /// An optimization for an inference session to have constant weights, true by
  /// default. Set this option to false if you are going to want to change the
  /// weights with a call to Session::resetHostWeights after the session has
  /// been prepared. This option has no effect on a training session
  bool constantWeights = true;

  /// Enable Poplar executable caching.
  bool enableEngineCaching = false;

  /// Folder to save the \c poplar::Executable to.
  std::string cachePath = "session_cache";

  /// Throw an exception when floating point errors occur.
  bool enableFloatingPointChecks = false;

  /// Enable stochastic rounding. PopART will set the Poplar engine option
  /// "target.deterministicWorkers" to "true" if this option is set and to
  /// "false" if it is not set. You can override this behaviour by adding a
  /// value for "target.deterministicWorkers" to SessionOptions::engineOptions.
  bool enableStochasticRounding = false;

  // Temporary option (not public) to enable RNG management as it currently
  // results in a performance hit. Set to `true` to enable RNG state management
  // functionality. TODO(T48752): remove this option (and default to enabling
  // RNG state management) once T48402 is implemented.
  bool _enableRngStateManagement = false;

  /// Configuration settings for execution phases.
  ExecutionPhaseSettings executionPhaseSettings;

  /// Configuration setting for operations in the accumulate outer fragment.
  AccumulateOuterFragmentSettings accumulateOuterFragmentSettings;

  /// Enable explicit recomputation.
  bool explicitRecomputation = false;

  bool explicitPipeliningEnabled() const {
    return enablePipelining && useHostCopyOps && enableExplicitMainLoops;
  }

  bool implicitPipeliningEnabled() const {
    return enablePipelining && !(useHostCopyOps && enableExplicitMainLoops);
  }

  /**
   * A wrapper class for the #numIOTiles option that permits any int value and
   * has an 'unassigned' state.
   */
  class NumIOTiles {
  public:
    /// Constructor.
    NumIOTiles();
    /// Constructor.
    NumIOTiles(int numIOTiles);

    /// Compare with int.
    bool operator==(const int &rhs) const;

    /// Auto convert to int.
    operator int() const;

    /// Assign value using int.
    NumIOTiles &operator=(const int &x);

  private:
    int value              = 0;
    bool userAssignedValue = false;
  };

  /// Number of IPU tiles dedicated to IO.
  NumIOTiles numIOTiles;

  /// Enable zero-copy for subgraphs.
  bool aliasZeroCopy = false;

  /// Configuration setting for batch serialization.
  BatchSerializationSettings batchSerializationSettings;

  /// Configuration settings for the autodiff transform.
  AutodiffSettings autodiffSettings;

  /// Options to delay variable updates as much as possible.
  /// TODO: Remove with T19212
  bool delayVarUpdates = true;

  /// When #shouldDelayVarUpdates is true, the other ops in the proximity of the
  /// delayed var updates may inherit the -inf schedule priority used to delay
  /// the var updates. This is undesirable for some ops that consume gradients,
  /// as we would like to consume (and thus be able to recycle the memory of)
  /// those gradients as soon as possible. Two examples are HistogramOps when
  /// doing automatic loss scaling, and the AccumulateOps that accumulate
  /// the gradients when doing gradient accumulation.
  ///
  /// If true, if #shouldDelayVarUpdates is true, this option will cause the
  /// schedule priority of the above described ops to be re-overriden to +inf.
  /// TODO: Remove with T19212.
  bool scheduleNonWeightUpdateGradientConsumersEarly = false;

  // TODO: Remove with T19212
  bool shouldDelayVarUpdates() const;

  /// Enable the global #fullyConnectedPass option for matmuls.
  bool enableFullyConnectedPass = true;

  /// Enable/disable the serializing of matmuls.
  bool enableSerializedMatmuls = true;

  /// For partialsTypeMatMuls, possible values are defined by
  /// `fromString` in op/matmul.cpp. As of last check, those are:
  /// "float", "half" in any letter case.

  /// Set the partials type globally for matmuls. Can be overridden individually
  /// with Builder.setPartialsType(). Valid values are `"float"` and `"half"`.
  /// By default, this is not set, so no global partials type is imposed.
  std::string partialsTypeMatMuls;

  /// If true, computes the mean first and subtracts the activations
  /// from it before computing the variance. The implementation with
  /// this flag set to true is slower than when set to false.
  /// The stable version requires the first order moment to be
  /// estimated and applied to the sample set before the second
  /// order central moment is calculated.
  bool enableStableNorm = false;

  /// Poplar engine options.
  std::map<std::string, std::string> engineOptions;

  /// Poplar convolution options.
  std::map<std::string, std::string> convolutionOptions;

  /// Poplar LSTM options.
  std::map<std::string, std::string> lstmOptions;

  std::map<std::string, std::string> matmulOptions;

  /// Poplar reporting options.
  std::map<std::string, std::string> reportOptions;

  /// GCL options
  std::map<std::string, std::string> gclOptions;

  /// List of codelets (with filetype) to be added to the Poplar graph. See the
  /// Poplar documentation for more information.
  std::vector<std::string> customCodelets;

  /// Compile flags for the custom codelets. For example `-g` to generate debug
  /// info.
  std::string customCodeletCompileFlags;

  /// The maximum allowed time that can be spent searching for a good graph
  /// schedule before a solution must be returned.
  double timeLimitScheduler = 1e9;

  /// The maximum number of improving steps allowed by the scheduling algorithm
  /// before a solution must be returned.
  int64_t swapLimitScheduler = static_cast<int64_t>(1e9);

  /// PopART uses Poprithms for scheduling PopART graphs. The Poprithms graphs
  /// created for scheduling can be optionally serialised (written to file). The
  /// string below specified the directory to serialize Poprithms graphs to. If
  /// it is empty, then the graphs will not be serialised. The names of
  /// serialization files will be `poprithms_shift_graph_i.json` for the lowest
  /// non-existing values of `i`. The directory must already exist, PopART will
  /// not create it.
  std::string serializedPoprithmsShiftGraphsDir{};

  /// The initial scheduling is done with Kahn's algorithm. When several Ops are
  /// free to be scheduled, this controls which method is used.
  std::string kahnTieBreaker = "greedy";

  /// The transitive closure optimization pass can significantly accelerate the
  /// scheduler. It does not in general affect the final schedule returned. It
  /// is run between initialization with Kahn's algorithms and the shifting
  /// swaps. The transitive closure optimization pass is O(nOps^2) and so should
  /// not be used for extremely large Graphs. If a Graph is above the following
  /// threshold, the transitive closure optimization pass is not run.
  size_t transitiveClosureOptimizationThreshold{100000};

  /// Replaces single sums of partial gradients with a tree of additions.
  /// This can reduce max liveness at the cost of extra cycles. A typical
  /// use case for this would be if a large weight tensor is used as an
  /// input to many operations.
  bool decomposeGradSum = false;

  /// Enable training with Poplar replicated graphs across multiple PopART
  /// instances.
  bool enableDistributedReplicatedGraphs = false;

  /// The total number of replicas in a multi instance replicated graph training
  /// session (this should be left as the default value (1) if distributed
  /// replicated graphs are disabled). This value includes local replication.
  int64_t globalReplicationFactor = 1;

  /// The first replica index that this PopART instance is running.
  int64_t globalReplicaOffset = 0;

  /// Helper method to handle the different replication options.
  /// If enableDistributedReplicatedGraphs is true
  ///   return globalReplicationFactor
  /// if enableReplicatedGraphs
  ///   return replicatedGraphCount
  /// otherwise
  ///   return 1
  int64_t getGlobalReplicationFactor() const;

  /// Helper method to check the accumulation factor settings for consistency
  /// if gradient accumulation is not enabled and the factor is set to >1.
  /// Returns the accumulation factor otherwise.
  unsigned getAccumulationFactor() const;

  /// Allows to group the streams from host at the beginning and the streams
  /// to host at the end, this trades off sum-liveness efficiency for cycle
  /// efficiency.
  bool groupHostSync = false;

  /// Strict op version checks will throw an error if the exact version of an op
  /// required for the models opset is not supported. Turning this check off
  /// will cause PopART to fall back to the latest implementation of the op that
  /// is supported. Warning, turning off these checks may cause undefined
  /// behaviour.
  bool strictOpVersions = true;

  /// Run Opx checks to verify IR tensor aliasing information
  /// corresponds to lowered Poplar tensor aliasing.
  bool opxAliasChecking = false;

  /// Run Opx checks to verify IR tensor modification information
  /// corresponds to lowered Poplar tensor modifications.
  bool opxModifyChecking = false;

  /// Uses IR graph operations for data and anchor streams
  bool useHostCopyOps = false;

  /// Allows to load/offload device RNG state from host.
  bool enableLoadAndOffloadRNGState = false;

  /// Tensor location settings for activation/gradient tensors.
  TensorLocationSettings activationTensorLocationSettings =
      TensorLocationSettings{TensorLocation(), 2, 8192};
  /// Tensor location for weight tensors.
  TensorLocationSettings weightTensorLocationSettings =
      TensorLocationSettings{TensorLocation(), 2, 8192};
  /// Tensor location for optimizer state tensors.
  TensorLocationSettings optimizerStateTensorLocationSettings =
      TensorLocationSettings{TensorLocation(), 2, 8192};
  /// Tensor location for gradient accumulator tensors.
  TensorLocationSettings accumulatorTensorLocationSettings =
      TensorLocationSettings{TensorLocation(), 2, 8192};

  /// Override tensor location for specific tensors by setting a TensorLocation
  /// for specific TensorId values.
  std::map<TensorId, TensorLocation> tensorLocationSettingsOverride;

  /// Settings to enable and configure the automatic loss scaling behaviour when
  /// training.
  ///
  /// **Note:** Automatic loss scaling is currently experimental and under
  /// active development. We recommend that the user sets the loss scale
  /// manually.
  AutomaticLossScalingSettings automaticLossScalingSettings;

  // Settings for developers to configure testing and benchmarking
  DeveloperSettings developerSettings;

  /// If enabled, casts any tensor of unsupported data types to supported data
  /// types when lowering to Poplar
  /// Currently, this implies casting:
  /// INT64 -> INT32
  /// UINT64 -> UINT32
  /// The cast will error for incompatible data types and over/underflows, and
  /// inform on narrowing casts
  bool enableSupportedDataTypeCasting = true;

  /// Enables explicit main loop transformation, and disables implicit training
  /// loops. This will become deprecated and enabled by default.
  bool enableExplicitMainLoops = false;

  /// Group norms have a fast math mode /which changes the implementation to run
  /// faster on IPU but as a consequence/ is incompatable with other
  /// implementations (i.e running trained weights on host). We default to
  /// correct and slightly slower but a user can opt into fast but incorrect.
  bool groupNormStridedChannelGrouping = false;

  /// Get the buffering depth for a TensorId. Will return 1 unless
  /// prefetching is enabled and the buffering depth is overwritten
  /// in the \c prefetchBufferingDepthMap variable.
  ///
  /// **Not part of public API**
  unsigned getPrefetchBufferingDepth(const TensorId &id,
                                     unsigned defaultValue) const;

  /// Callback function used to to indicate
  /// PopART compilation progress. The function is
  /// passed two integers. The first is the progress
  /// value and the second is the maximum value for
  /// the progress.
  ///
  /// The function should not block. All calls
  /// to the callback function will be made from the main thread so
  /// blocking in the callback will block compilation from progressing.
  ///
  /// If this logger is not set then compilation progress will be
  /// printed on the info channel.
  std::function<void(int, int)> compilationProgressLogger;

  /// Total progress ticks until compilation complete
  int compilationProgressTotal = 100;

  /// Returns true if auto-recomputation is enabled.
  bool autoRecomputationEnabled() const;

  /// Enables merging remote and host IO operations to facilitate IO overlap
  bool enableMergeExchange = true;

  /// Only compatible with models that have an fp16 loss scale tensor. When
  /// `true` the loss scale tensor will be an fp32 tensor, and will be combined
  /// with fp16 activations as late as possible to produce the first fp16
  /// activation gradients. This allows the user to choose a loss scale value
  /// greater than max(fp16). This is also recommended when automatic loss
  /// scaling is enabled.
  bool ensureFp32LossScaleTensor = false;

  SessionOptions() {
    // Automatically set `enableEngineCaching` and `cachePath` if the
    // environment variable `POPART_CACHE_DIR` is provided
    auto cachePathEnv = getPopartEnvVar("CACHE_DIR");
    if (cachePathEnv) {
      enableEngineCaching = true;
      cachePath           = *cachePathEnv;
    }
  }
};

} // namespace popart

namespace std {
template <> struct hash<popart::SessionOptions> {
  std::size_t operator()(const popart::SessionOptions &so) const;
};
} // namespace std

#endif
