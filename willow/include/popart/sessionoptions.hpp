// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SESSIONOPTIONS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SESSIONOPTIONS_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <popart/op.hpp>
#include <popart/tensorlocation.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"
#include "popart/util.hpp"

// Note that comments in this file have to adhere to doxygen formatting. See
// https://www.doxygen.nl/manual/.

namespace popart {

/**
 * Enum type to specify the method for selecting gradient tensors whose
 * statistics are to be tracked for the AutomaticLossScale transform.
 */
enum class GradientTensorTrackingMethod {
  /// Track all gradients of non-view-changing gradient tensors.
  AllNonViewChangingGradientTensors = 0,
  /// Track all gradients of inputs to MatMul and Convolution ops.
  ConvAndMatmulGradients,
  /// Track gradients of user-specified tensors.
  GradientsOfUserSpecifiedTensors,
  /// The number of \c GradientTensorTrackingMethod values.
  N
};

/**
 * A structure containing user configuration for automatic loss scaling
 * settings.
 *
 * \note Automatic loss scaling is in preview. It is well tested
 * and enabled in some of our example applications, but may not behave
 * as expected in all models. Recommendation: if your model with
 * automatic loss scaling enabled does not converge or triggers a
 * compilation error, then you will need to set the loss scale manually.
 */
struct AutomaticLossScalingSettings {
  /// Default constructor for AutomaticLossScalingSettings.
  AutomaticLossScalingSettings() = default;

  /**
   * Constructor for AutomaticLossScalingSettings.
   * \param enabled_ Indicate whether to keep track (`true`) or not (`false`) of
   *        the distribution of gradient tensor elements over the floating point
   *        range. Default: `false`.
   * \param toTrackTensors_ An optional list of model tensor names, for which
   *        gradient statistics will be collected. If not set, the gradients of
   *        all tensors produced by default operations (matmul, conv) will
   *        be used.
   * \param binEdgeLocation_ The location of the bin edge as a proportion of the
   *        absolute numerical range of the tracked gradient tensor elements, in
   *        the range [0, 1]. 0 represents the smallest representable value,
   *        and 1 the maximum. This is the single bin edge of the histogram
   *        that is an input to the loss scale updater algorithm. Default:
   *        0.125.
   * \param thresholdUpperCountProportion_ The proportion of the elements
   *        in the upper bin above which the loss scale is increased, and below
   *        which the loss scale is decreased. Should be in the range [0, 1].
   *        Default: 1e-7.
   * \param updatePeriod_ Indicate how often the loss scale update factor should
   *        be updated with respect to optimizer steps. Default: 1
   * \param gradientTensorTrackingMethod_ The method for selecting gradient
   *        tensors whose statistics are to be tracked. Default:
   *        GradientTensorTrackingMethod::AllNonViewChangingGradientTensors.
   */
  AutomaticLossScalingSettings(
      bool enabled_,
      const nonstd::optional<std::vector<TensorId>> &toTrackTensors_,
      float binEdgeLocation_,
      float thresholdUpperCountProportion_,
      int updatePeriod_,
      GradientTensorTrackingMethod gradientTensorTrackingMethod_);

  std::size_t hash() const;
  /*
   * Indicate whether to keep track of the distribution of gradient tensor
   * elements over the floating point range.
   * Adjust the value loss scaling tensor accordingly, with the aim of
   * preventing underflow or overflow.
   * If `true`, keeps track of the distribution, `false` if not.
   */
  bool enabled = false;

  /*
   * The location of the bin edge as a proportion of the absolute numerical
   * range of the tracked gradient tensor elements, in the range [0, 1]. 0
   * represents the smallest representable value, and 1 the maximum. This is
   * the single bin edge of the histogram that is an input to the loss scale
   * updater algorithm. Default: 0.125.
   */
  float binEdgeLocation = 0.125f;

  /*
   * The proportion of the elements in the upper bin above which the loss scale
   * is increased, and below which the loss scale is decreased. Should be in
   * the range [0, 1]. Default: 1e-7.
   */
  float thresholdUpperCountProportion = 1e-7;

  /*
   * An optional list of model tensor names, for which gradient statistics
   * will be collected. If not set, the gradients of all tensors produced
   * by default operations (matmul, conv) will be used.
   */
  nonstd::optional<std::vector<TensorId>> toTrackTensors;

  /*
   * Indicate how often the loss scale update factor should be updated with
   * respect to optimizer steps. Default: 1.
   */
  int updatePeriod = 1;

  /*
   * The method for selecting gradient tensors whose statistics are to be
   * tracked. Default:
   * GradientTensorTrackingMethod::AllNonViewChangingGradientTensors.
   */
  GradientTensorTrackingMethod gradientTensorTrackingMethod =
      GradientTensorTrackingMethod::AllNonViewChangingGradientTensors;
};

/**
 * Enum type to specify which ops to recompute in the backward pass when doing
 * auto-recomputation.
 */
enum class RecomputationType {
  /// No ops are recomputed (Default).
  None = 0,
  /**
   * Recompute using algorithm that picks checkpoints to try and minimise max
   * liveness.
   */
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
 * Enum type used to specify which VarUpdateOp ops to merge.
 */
enum class MergeVarUpdateType {
  /// Do not merge VarUpdateOp ops.
  None = 0,
  /**
   * Merge all VarUpdateOp ops into as few groups as possible.
   * This is a good choice when memory is not a constraint.
   */
  All,
  /**
   * Merge into groups while attempting not to increase maximum
   * variable liveness, and also not slice tensor variables so
   * they will need to be processed by different VarUpdateOp ops.
   */
  AutoLoose,
  /**
   * Merge into groups, so that VarUpdateOp ops process tensors of
   * exactly \c SessionOptions::mergeVarUpdateMemThreshold in size.
   */
  AutoTight,
  /// The number of \c MergeVarUpdateType values.
  N
};

/// A structure containing settings for replicated collective operations.
struct ReplicatedCollectivesSettings {
  /**
   * Constructor for the ReplicatedCollectivesSettings struct.
   * \param prepareScheduleForMergingCollectives Insert constraints into the
   *       schedule such that collectives which can be merged occur one right
   *       after the other. `true` to insert constraints, `false` otherwise.
   *       Default: `false`.
   * \param mergeAllReduceCollectives Identify allreduce operations which can be
   *       scheduled at the same time, and perform them as one larger operation
   *       to better utilize the bandwidth between replicas. `true` to identify
   *       operations, `false` otherwise. Default: `false`.
   */
  ReplicatedCollectivesSettings(
      bool prepareScheduleForMergingCollectives = false,
      bool mergeAllReduceCollectives            = false,
      bool mergeReduceScatterCollectives        = false,
      bool mergeAllGatherCollectives            = false);

  std::size_t hash() const;

  /*
   * Insert constraints into the schedule such that collectives
   * which can be merged occur one after the other.
   * `true` to insert constraints, `false` otherwise.  Default: `false`.
   */
  bool prepareScheduleForMergingCollectives = false;

  /*
   * Identify allreduce operations which can be scheduled
   * at the same time, and perform them as one larger operation
   * to better utilize the bandwidth between replicas.
   * `true` to identify operations, `false` otherwise. Default: `false`.
   */
  bool mergeAllReduceCollectives = false;

  /// Identifies reduce-scatter operations which can be scheduled
  /// at the same time, and performs them as one larger operation
  /// so as to better utilize the bandwidth between replicas
  bool mergeReduceScatterCollectives = false;

  /// Identifies allgather operations which can be scheduled
  /// at the same time, and performs them as one larger operation
  /// so as to better utilize the bandwidth between replicas
  bool mergeAllGatherCollectives = false;
};

/// Enum type used to specify a virtual graph mode.
enum class VirtualGraphMode {
  /// Virtual graphs are not enabled.
  Off = 0,
  /// User must set the popart::Op::virtualGraph attribute on all ops.
  Manual,
  /// Use the AutoVirtualGraph transform.
  Auto,
  /// Virtual graphs are tied to execution phases.
  ExecutionPhases,
  /// The number of \c VirtualGraphMode values.
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
  /// Input tensors are initialised with a random normal distribution ~N(0,1).
  RandomNormal,
  /// Input tensors are initialised with a uniform distribution.
  RandomUniform,
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

/// Return a string value for VirtualGraphMode.
std::string toString(VirtualGraphMode);
/// Stream the value for VirtualGraphMode.
std::ostream &operator<<(std::ostream &, VirtualGraphMode);

/// Return a string value for RecomputationType.
std::string toString(RecomputationType);
/// Stream the value for RecomputationType.
std::ostream &operator<<(std::ostream &, RecomputationType);

/**
 * A structure containing user configuration for cache/offloading settings.
 */
struct TensorLocationSettings {

  /// Constructor.
  TensorLocationSettings() = default;

  /**
   * Constructor.
   * \param location_ The tensor location information.
   * \param minElementsForOffChip_ The minimum number of elements below which
   *        offloading won't be considered.
   * \param minElementsForReplicatedTensorSharding_ The minimum number of
   *        elements necessary for replicated tensor sharding.
   */
  TensorLocationSettings(TensorLocation location_,
                         int minElementsForOffChip_                  = 2,
                         int minElementsForReplicatedTensorSharding_ = 8192);

  /**
   * Constructor.
   * \param storage_ The tensor storage information.
   * \param minElementsForOffChip_ The minimum number of elements below which
   *        offloading won't be considered.
   * \param minElementsForReplicatedTensorSharding_ The minimum number of
   *        elements necessary for replicated tensor sharding.
   */
  TensorLocationSettings(TensorStorage storage_,
                         int minElementsForOffChip_                  = 2,
                         int minElementsForReplicatedTensorSharding_ = 8192);

  /// The default tensor location for this tensor type.
  TensorLocation location = TensorLocation();

  /// The minimum number of elements below which offloading won't be considered.
  int minElementsForOffChip = 2;

  /**
   * A minimum number of elements below which replicated tensor sharding
   * won't be considered.
   */
  int minElementsForReplicatedTensorSharding = 8192;
};

/// Return a string value for TensorLocationSettings.
std::string toString(const TensorLocationSettings &);
/// Stream the value for TensorLocationSettings.
std::ostream &operator<<(std::ostream &, const TensorLocationSettings &);

/**
 * Enum type that describes how to change the batch serialisation subgraph
 * schedule before outlining.
 * \note This setting is experimental and may change.
 */
enum class BatchSerializationBatchSchedule {
  /**
   * Don't encourage any particular scheduling for ops within batch subgraphs
   * (leave it to the scheduler) but tell the scheduler to schedule subgraphs
   * in sequence.
   */
  Scheduler = 0,
  /**
   * Encourage all ops within batch subgraphs to be scheduled identically and
   * for each subgraph to be scheduled in sequence (good for outlineability).
   */
  Isomorphic,
  /**
   * Attempt to put the remote load op for batch N+1 right after the
   * compute phase of batch N.
   */
  OverlapOnIo,
  /**
   * Attempt to put the remote load op for batch N+1 right before
   * the compute phase of batch N.
   */
  OverlapOnCompute,
  /// The number of \c BatchSerializationBatchSchedule values.
  N
};

/**
 * Enum type that describes when to apply batch serialization.
 * \note This setting is experimental and may change.
 */
enum class BatchSerializationTransformContext {
  /// Apply batch serialiation before growing the backward pass.
  Fwd = 0,
  /// Apply batch serialiation after growing the backward pass.
  Bwd,
  /// The number of \c BatchSerializationTransformContext values.
  N
};

/**
 * Enum type that describes how to apply the batch serialization.
 * \note This setting is experimental and may change.
 */
enum class BatchSerializationMethod {
  /// Unroll the batch with dynamic slicing.
  UnrollDynamic = 0,
  /// Unroll the batch with static slicing.
  UnrollStatic,
  /// Loop over the batch dimension.
  Loop,
  /// The number of \c BatchSerializationMethod values.
  N
};

/**
 * A structure containing batch serialization settings.
 */
struct BatchSerializationSettings {
  /// Default constructor for BatchSerializationSettings.
  BatchSerializationSettings() = default;

  /**
   * Constructor for BatchSerializationSettings.
   *
   * \param factor_ The number of compute batches to split operations into.
   *        Default: 0.
   * \param concatOnVirtualGraphChange_ Indicate to break batch serialization
   *        chains (`true`) when the virtual graph changes (by concatenating the
   *        compute batches to the local batch). Default: `true`.
   * \param concatOnExecutionPhaseChange_ Indicate to break batch serialization
   *        chains (`true`) when the execution phase changes (by concatenating
   *        the compute batches to the local batch). Default: `true`.
   * \param concatOnPipelineStageChange_ Indicate to break batch serialization
   *        chains (`true`) when the pipeline stage changes (by concatenating
   *        the compute batches to the local batch). Default: `true`.
   * \param transformContext_ An experimental value to control when batch
   *        serialization is applied. Default: ::Fwd.
   * \param method_ An experimental value to control how batch serialization is
   *        applied. Default: BatchSerializationMethod::UnrollDynamic.
   * \param batchSchedule_ An experimental value that changes how operations are
   *        scheduled. Default: BatchSerializationBatchSchedule::Isomorphic.
   */
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

  /// The number of compute batches to split operations into.
  int factor = 0;
  /**
   * Break batch serialization chains when the virtual graph
   * changes (by concatenating the compute batches to the local batch).
   */
  bool concatOnVirtualGraphChange = true;
  /**
   * Break batch serialization chains when the execution phase
   * changes (by concatenating the compute batches to the local batch).
   */
  bool concatOnExecutionPhaseChange = true;
  /**
   * Break batch serialization chains when the pipeline stage
   * changes (by concatenating the compute batches to the local batch).
   */
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
 * The steps for phased execution are:
 *
 * -# Copy to IO tiles if necessary.
 * -# Run collective operations if necessary.
 * -# Load optimizer state.
 * -# Update optimizer state.
 * -# Apply optimizer.
 * -# Store updated tensor if necessary.
 */
enum class ExecutionPhaseSchedule {
  /**
   * Process above steps for one weight at a time (for example: 123456, 123456,
   * 123456). The scheduler may interleave these steps.
   */
  Interleaving = 0,
  /**
   * Process above steps for all weights together, in a way that maximises
   * overlap potential between compute and exchange
   * (for example: 333, 111, 222, 444, 555, 666).
   */
  Batch,
  /**
   * Process above steps for all weights together, in a way that maximises
   * overlap potential between compute and exchange, and maximise stream
   * copy merges by keeping RemoteLoad/RemoteStore operations clustered
   * (for example: 333, 111, 222, 444, 555, 666).
   */
  BatchClusteredIO,
  /// The number of \c ExecutionPhaseSchedule values.
  N
};

/**
 * A structure containing ExecutionPhase settings.
 */
struct ExecutionPhaseSettings {
  /// Default constructor for ExecutionPhaseSettings.
  ExecutionPhaseSettings() = default;

  /**
   * Constructor for ExecutionPhaseSettings.
   *
   * \param phases_ The number of execution phases for the whole model.
   *      Default=0.
   * \param stages_ The number of overlapping stages:
   *      * 1: Parallel streaming memory, default for 1 IPU per replica.
   *      * 2: PingPong between 2 IPUs, default for 2 or more IPUs per replica
   *           (Default).
   * \param weightIOSchedule_ The execution phase IO schedule for weight
   *      tensors. Default: ExecutionPhaseIOSchedule::Preload.
   * \param activationIOSchedule_ The execution phase IO schedule
   *      for activation and gradient tensors. Default:
   *      ExecutionPhaseIOSchedule::Preload.
   * \param optimizerStateIOSchedule_ An experimental value to control
   *      when batch serialization is applied. Default:
   *      ExecutionPhaseIOSchedule::OnDemand.
   * \param accumulatorIOSchedule_ An experimental value to control how
   *      batch serialization is applied. Default:
   *      ExecutionPhaseIOSchedule::Preload.
   * \param schedule_ An experimental value that changes how operations are
   *      scheduled. Default: ExecutionPhaseSchedule::Interleaving.
   */
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

  /// Number of ExecutionPhases for the whole model.
  int phases = 0;

  /**
   * Number of overlapping stages
   *  * 1: Parallel streaming memory, default for 1 IPU per replica.
   *  * 2: PingPong between 2 IPUs, default for 2 or more IPUs per replica.
   */
  int stages = 2;

  /// The execution phase IO schedule for weight tensors.
  ExecutionPhaseIOSchedule weightIOSchedule = ExecutionPhaseIOSchedule::Preload;
  /// The execution phase IO schedule for activation and gradient tensors.
  ExecutionPhaseIOSchedule activationIOSchedule =
      ExecutionPhaseIOSchedule::Preload;

  // TODO T28528: Add doxygen comments.
  ExecutionPhaseIOSchedule optimizerStateIOSchedule =
      ExecutionPhaseIOSchedule::OnDemand;
  ExecutionPhaseIOSchedule accumulatorIOSchedule =
      ExecutionPhaseIOSchedule::Preload;
  ExecutionPhaseSchedule schedule = ExecutionPhaseSchedule::Interleaving;
};

/**
 * Enum type that determines how the operations in the accumulate outer fragment
 * will be scheduled across virtual graphs (only relevant to pipelined modes).
 */
enum class AccumulateOuterFragmentSchedule {
  /// Don't add additional constraints and let the scheduler work it out.
  Scheduler = 0,
  /// Add constraints that ensure ops are executed in virtual graph ID order.
  Serial,
  /**
   * Try and parallelise ops with different virtual graph IDs as much as
   * possible.
   */
  OverlapCycleOptimized,
  /**
   * Try and parallelise ops with different virtual graph IDs but avoid certain
   * steps that are costly in terms of memory usage.
   */
  OverlapMemoryOptimized
};

/**
 * A structure containing accumulate outer fragment settings.
 */
struct AccumulateOuterFragmentSettings {
  // Default constructor for AccumulateOuterFragmentSettings.
  AccumulateOuterFragmentSettings() = default;

  /**
   * Constructor for AccumulateOuterFragmentSettings.
   * \param schedule_ Indicate how to schedule the accumulate outer fragment.
   *        This setting is experimental and may change. Default:
   *        AccumulateOuterFragmentSchedule::Serial
   * \param excludedVirtualGraphs_ Indicate to explicitly avoid parallelising
   *        the virtual graph IDs. This setting is experimental and may change.
   */
  AccumulateOuterFragmentSettings(
      AccumulateOuterFragmentSchedule schedule_,
      const std::vector<int> &excludedVirtualGraphs_)
      : schedule{schedule_}, excludedVirtualGraphs{excludedVirtualGraphs_} {}

  /**
   * Indicate how to schedule the accumulate outer fragment.
   * \note This setting is experimental and may change.
   */
  AccumulateOuterFragmentSchedule schedule =
      AccumulateOuterFragmentSchedule::Serial;
  /**
   * Indicate to explicitly avoid parallelising the virtual graph IDs.
   * \note This setting is experimental and may change.
   */
  std::vector<int> excludedVirtualGraphs = {};
};

/// Return a string value for AccumulateOuterFragmentSchedule.
std::string toString(AccumulateOuterFragmentSchedule r);
/// Stream the value for AccumulateOuterFragmentSchedule.
std::ostream &operator<<(std::ostream &out,
                         AccumulateOuterFragmentSchedule aofSched);

/**
 * Enum type that describes how copies for inputs and outputs for subgraphs
 * are lowered. Currently this only affects subgraphs associated with CallOp
 * ops.
 */
enum class SubgraphCopyingStrategy {
  /**
   * Copy all inputs before the start of the subgraph, copy all outputs after
   * all ops in the subgraph. With this strategy, subgraphs will always map
   * to a single Poplar function.
   */
  OnEnterAndExit = 0,
  /**
   * Copy inputs just before they are consumed and copy outputs as soon as
   * they are produced. With this strategy, subgraphs may be lowered into
   * multiple Poplar functions.
   */
  JustInTime,
  /// The number of \c SubgraphCopyingStrategy values.
  N
};

/**
 * Enum type that specifies when to divide by a mean reduction factor, when
 * doing mean reduction over a sequence of tensors \f$t_1, t_2, ..., t_k\f$.
 */
enum class MeanReductionStrategy {
  /**
   * Keep the reduction buffer as the mean of the tensors accumulated so far.
   * If \f$t_1, ..., t_f\f$ has just been processed,
   * the current accumulator \f$s\f$ is the mean of these values, and
   * the next accumulator update is
   * \f$s = \frac{f}{f+1} * s + \frac{1}{f+1} * t_{f+1}\f$ to keep \f$s\f$ a
   * running mean.
   *
   * This strategy guarantees \f$s \le \max(a_1, ..., a_k)\f$ throughout the
   * accumulation, therefore it will not overflow, but it is generally slower
   * than MeanReductionStrategy::Post.
   */
  Running = 0,
  /**
   * Keep the accumulation factor as the running sum,
   * and divide once by \f$k\f$ at the end of the accumulation.
   * This strategy will generally be faster than MeanReductionStrategy::Running,
   * but is prone to overflow (especially when using `fp16`).
   */
  Post,
  /// The number of \c MeanReductionStrategy values.
  N
};

/**
 * Enum type representing a strategy to ensure a backward graph's inputs are
 * either inputs of the forward graph, outputs of the forward graph or
 * gradients of outputs of the forward graph. Strategies may expose tensors
 * that would otherwise have been internal to the forward graph as outputs of
 * this forward graph.
 */
enum class AutodiffStitchStrategy {
  /**
   * Recompute any backward graph inputs associated with non-gradient forward
   * graph tensors that are neither inputs nor outputs in the forward graph.
   */
  RecomputeMinimal = 0,
  /**
   * Recompute any backward graph inputs associated with non-gradient forward
   * graph tensors that are not inputs in the forward graph.
   */
  RecomputeAllNonInputs,
  /**
   * For backward graph inputs associated with non-gradient forward graph
   * tensors that are neither inputs or outputs in the forward graph, add them
   * as outputs to the forward graph.
   *
   * \note This strategy is not guaranteed to work for all circumstances. In
   * particular, it is unable to deal with subgraphs of IfOp. Using this
   * setting may therefore result in subsequent exceptions in the Autodiff
   * transform and it is therefore inadvisable to use this as an Autodiff
   * default.
   */
  AddFwdOutputs,
  /**
   * Like AutodiffStitchStrategy::AddFwdOutputs except that those backward graph
   * inputs that can't be stitched with AutodiffStitchStrategy::AddFwdOutputs
   * (that is, by adding outputs to the forward graph) are stitched using the
   * AutodiffStitchStrategy::RecomputeMinimal strategy instead. This
   * means that this is a safe strategy to use as an Autodiff default.
   */
  SafeAddFwdOutputs,
  /// Number of \c AutodiffStitchStrategy values.
  N
};
std::string toString(const AutodiffStitchStrategy &);
std::ostream &operator<<(std::ostream &, const AutodiffStitchStrategy &);

/**
 * The settings for the Autodiff transform.
 */
struct AutodiffSettings {
  /// Default constructor for the AutodiffSettings struct.
  AutodiffSettings() = default;

  /**
   * Constructor for the AutodiffSettings struct.
   * \param stitchStrategy_ The strategy to ensure a backward graph's inputs are
   *        either inputs of the forward graph, outputs of the forward graph or
   *        gradients of outputs of the forward graph. Default:
   *        AutodiffStitchStrategy::RecomputeAllNonInputs.
   */
  AutodiffSettings(AutodiffStitchStrategy stitchStrategy_)
      : stitchStrategy{stitchStrategy_} {}

  /**
   * The strategy PopART should use to ensure that all graph inputs of a
   * backward graph are available as either inputs or outputs of the forward
   * graph or gradients of outputs of the forward graph.
   *
   * \note This is an experimental option and may change.
   */
  AutodiffStitchStrategy stitchStrategy =
      AutodiffStitchStrategy::RecomputeAllNonInputs;
};

/// Struct for development-specific configuration intended to be used by
/// PopART developers, as opposed to PopART users.
///
/// NOTE: These options are not subject to deprecation notices and may be
/// changed or removed at any time.
struct DeveloperSettings {
  /// The minimum percentage of the total time a scope must take in order
  /// for it to be logged by the TimePartitionLogger.
  double timePartitionLoggerThresholdPercentage = 1.0f;
};

/**
 * A structure containing user configuration options for the Session class.
 */
struct SessionOptions {
  /// A directory for log traces to be written into.
  std::string logDir;

  /// When to write `.dot` files during IR construction.
  std::set<std::string> dotChecks = {};

  /**
   * The ops written to the `.dot` file will be a part
   * of the schedule, controlled by firstDotOp and finalDotOp. In particular,
   * it will be [max(0, firstDotOp), min(N ops in IR, finalDotOp)).
   */
  int firstDotOp = 0;
  /// See firstDotOp.
  int finalDotOp = 10000;

  /**
   * Enable inclusion of the op name in the `.dot` file (the op type is always
   * exported).
   * Enabled when `true`. Default: `false`.
   */
  bool dotOpNames = false;

  /**
   * Enable export of Poplar computational graph.
   * Enabled when `true`. Default: `false`.
   */
  bool exportPoplarComputationGraph = false;

  /**
   * Enable export of Poplar vertex graph.
   * Enabled when `true`. Default: `false`.
   */
  bool exportPoplarVertexGraph = false;

  /**
   * Enable creation of separate PDFs for each subgraph when generating PDFs of
   * IR graphs.
   * Enabled when `true`. Default: `true`.
   */
  bool separateCallOpPdfs = true;

  /**
   * Enable outlining.
   * This identifies and extracts repeated parts of computational graph into
   * subgraphs.
   * Enabled when `true`. Default: `true`.
   */
  bool enableOutlining = true;

  /**
   * Enable inclusion of the cost of copying of cached sections should be
   * in the outlining cost model.
   * Enabled when `true`. Default: `true`.
   */
  bool enableOutliningCopyCostPruning = true;

  /**
   * Specify the incremental value that a sub-graph requires, relative to its
   * nested sub-graphs (if any), to be eligible for outlining.
   *
   * A high threshold
   * results in fewer sub-graphs being outlined, a negative value results in all
   * being outlined. The gross value of a sub-graph is the sum of its
   * constituent ops' Op::getSubgraphValue() values. To disable outlining, it is
   * better to set enableOutlining to false than to set this value to infinity.
   * The default value of 1.0f results in all high value operations such as
   * convolution being cached, but standalone low value operations such as ReLU
   * will not be.
   *
   * Default: 1.0f.
   */
  float outlineThreshold = 1.0f;

  /**
   * Specify the penalty applied to outlining potential sub-graphs if the
   * sub-graph to be created breaks up a sequence of operations that are more
   * efficient (for example for overlapping compute and exchange) when outlined
   * together.
   *
   * Default: 10000.0f.
   */
  float outlineSequenceBreakCost = 10000.0f;

  /**
   * Specify how copies for inputs and outputs for subgraphs
   * are lowered.
   *
   * Setting this value to SubgraphCopyingStrategy::JustInTime may save
   * memory at the cost of fragmenting subgraphs into multiple Poplar functions.
   * This may be particularly useful when a number of weight updates are
   * outlined in one subgraph, as it may prevent multiple weight tensors from
   * being live at the same time inside the subgraph.
   *
   * Default: SubgraphCopyingStrategy::OnEnterAndExit.
   */
  SubgraphCopyingStrategy subgraphCopyingStrategy =
      SubgraphCopyingStrategy::OnEnterAndExit;

  /**
   * Enable recomputation of operations in the graph in the backward pass.
   * This will reduce model size at the cost of computation cycles.
   *
   * Default: RecomputationType::None (no recomputation).
   */
  RecomputationType autoRecomputation = RecomputationType::None;

  /**
   * Enable merging of VarUpdates into groups of VarUpdates, by flattening
   * and concatenating variable tensors and updating tensors.
   *
   * Default: MergeVarUpdateType::None (no merging).
   */
  MergeVarUpdateType mergeVarUpdate = MergeVarUpdateType::None;

  /**
   * Specify the memory threshold for VarUpdateOp merging algorithms.
   *
   * The MergeVarUpdateType::AutoLoose and MergeVarUpdateType::AutoTight
   * VarUpdateOp merging algorithms have a threshold on the total memory of
   * variable tensors to merge for updating. Defined as total memory in bytes.
   *
   * Default: 1000000.
   */
  int64_t mergeVarUpdateMemThreshold = 1000000;

  /**
   * Specify the threshold at peak used in the calculation of the absolute
   * threshold in the MergeVarUpdateType::AutoLoose VarUpdateOp merging
   * algorithm.
   *
   *  ```{.py}
   *  min(mergeVarUpdateMemThreshold, liveAtPeak - liveCurrently +
   * looseThresholdAtPeak)
   * ```
   * where:
   *  * \c liveAtPeak is an estimate of the maximum live memory of the
   *    computation; and
   *  * \c liveCurrently is an estimate of the live memory where the
   *    threshold is being used to determine whether to schedule or postpone a
   *    VarUpdateOp.
   *
   * Default: 80000.
   */
  int64_t looseThresholdAtPeak = 8000;

  /**
   * Enable rearrangement (in memory) of anchor tensors to be done on the host.
   *
   * Before anchor tensors are streamed from device to host, they are not
   * necessarily arranged in memory as required when they are to be copied
   * from host stream to host. This can be done on the device or on the host.
   *
   * Default: `true` (Rearrangement done on host to save memory, but often at
   * the expense of cycles, especially for larger anchor tensors.).
   */
  bool rearrangeAnchorsOnHost = true;

  /**
   * Enable rearrangement (in memory) of stream tensors to be done on the host.
   * Before stream tensors are streamed from host to device, they are not
   * necessarily arranged in memory as required when they are to be copied
   * from host stream to device. This can be done on the device or on the host.
   *
   * Default: `false` (Rearrangement done on device).
   */
  bool rearrangeStreamsOnHost = false;

  /**
   * Enable prefetching for input data streams.
   *
   * Poplar will speculatively read data for a stream before it is required in
   * order to allow the 'preparation' of the data to occur in parallel with
   * compute. Enabled when `true`. Default: `true`.
   */
  bool enablePrefetchDatastreams = true;

  /**
   * Specify the default buffering depth value used for streams that are not
   * re-arranged on the host.
   * For tensors that are rearranged on the host, a buffering
   * depth of 1 will always be used. This default value can be overridden via
   * bufferingDepthMap.
   */
  unsigned defaultBufferingDepth = 1;

  /**
   * \deprecated This session option name has been deprecated and will be
   * removed in a future release. Please use the alias defaultBufferingDepth
   * instead.
   */
  unsigned defaultPrefetchBufferingDepth =
      initialDefaultPrefetchBufferingDepthValue;

  /**
   * This mapping can be used to set stream-specific buffering depths.
   * The buffering depth could be thought of as being the size of a circular
   * buffer that feeds data to and from Poplar. A buffering depth greater than
   * 1 may improve the performance due to increased parallelisation but comes
   * at the cost of increasing the memory footprint. Streams for tensors that
   * have no entry in this map will default to 1 (if a tensor is rearranged on
   * host) or defaultBufferingDepth (if a tensor is not rearranged on host).
   * Specifying a tensor that gets rearranged on host in this map will throw an
   * error.
   */
  std::map<TensorId, unsigned> bufferingDepthMap;

  /**
   * \deprecated This session option name has been deprecated and will be
   * removed in a future release. Please use the alias bufferingDepthMap
   * instead.
   */
  std::map<TensorId, unsigned> prefetchBufferingDepthMap;

  /**
   * Enable the non-stable softmax Poplar function.
   *
   * By default, the stable softmax Poplar function is used. The input tensor
   * to softmax, \f$x\f$, is preprocessed by subtracting \f$max(x)\f$ from each
   * element
   * before computing the exponentials, ensuring numerical stability. If the
   * inputs to the softmax operations are small enough to not
   * cause overflow when computing the exponential, then the non-stable version
   * can be enabled instead, to increase the speed.
   *
   * Default: `false` (not enabled).
   */
  bool enableNonStableSoftmax = false;

  /// Enable replication of graphs. Default: `false` (not enabled).
  bool enableReplicatedGraphs = false;

  /// Enable gradient accumulation. Default: `false` (not enabled).
  bool enableGradientAccumulation = false;

  /**
   * Specify how gradients are reduced when using gradient accumulation
   * and graph replication. Default: ReductionType::Sum.
   */
  ReductionType accumulationAndReplicationReductionType = ReductionType::Sum;

  /**
   * Specify when to divide by a mean reduction factor when
   * accumulationAndReplicationReductionType is set to ReductionType::Mean.
   *
   * Default: MeanReductionStrategy::Post.
   */
  MeanReductionStrategy meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Post;

  /**
   * Specify the number of model replications.
   * If \c enableReplicatedGraphs is `true`, \c replicatedGraphCount will set
   * the number of model replications. For example, if the model uses 1 IPU, a
   * \c replicatedGraphCount of 2 will use 2 IPUs. If the model is
   * pipelined across 4 IPUs, a \c replicatedGraphCount of 4 will use 16 IPUs
   * in total. Therefore, the number of IPUs requested must be a multiple of
   * \c replicatedGraphCount. If the training is done across multiple instances
   * of the program then the \c replicatedGraphCount is the number of replicas
   * for this instance.
   */
  int64_t replicatedGraphCount = 1;

  /**
   * Specify the number of micro-batches to accumulate before applying the
   * varUpdate.
   */
  int64_t accumulationFactor = 1;

  /**
   * Specify how to place ops on virtual graphs to achieve model
   * parallelism, either manually using model annotations, or automatically.
   *
   * Default: VirtualGraphMode::Off.
   */
  VirtualGraphMode virtualGraphMode = VirtualGraphMode::Off;

  /**
   * Specify split ratios when VirtualGraphModel::Auto enabled.
   *
   * These values represent split ratios in each device and
   * each of the values is in range (0, 1).
   *
   * For example, to uniformly split the whole graph on 4 IPUs, the value should
   * be [0.25, 0.25, 025, 0.25].
   */
  std::vector<float> virtualGraphSplitRatios;

  /// Enable pipelining of virtual graphs. Default: `false` (not enabled).
  bool enablePipelining = false;

  /**
   * Specify whether to use real or synthetic data to initialize
   * input tensors.
   * Streaming to/from the host is only enabled for SyntheticDataMode::Off which
   * indicates that real data is being used.
   *
   * Default: SyntheticDataMode::Off.
   */
  SyntheticDataMode syntheticDataMode = SyntheticDataMode::Off;

  /**
   * Add instrumentation to the program to count the number of device cycles
   * (of a single tile, on a single IPU) that the main program takes to
   * execute. Expect this to have a small detrimental impact on performance.
   */
  bool instrumentWithHardwareCycleCounter            = false;
  std::set<Instrumentation> hardwareInstrumentations = {Instrumentation::Outer};

  /**
   * Disable saving of weight gradient tensors off the device.
   *
   * If `true`, the weight gradient tensors are not saved off the device
   * when \c devicex.weightsFromHost() is called.
   * \note This option is
   * overridden if \c syntheticDataMode is not SyntheticDataMode::Off.
   *
   * \note Weight gradient tensors that are also optimiser tensors will
   * only be disabled if both \c disableGradAccumulationTensorStreams and
   * \c disableOptimizerStateTensorStreams are `true`.
   */
  bool disableGradAccumulationTensorStreams = false;

  /**
   * Disable streaming of optimizer tensors.
   *
   * If `true`, streaming of optimizer tensors is disabled. This setting can be
   * used to conserve memory if you are not interested in checkpointing the
   * optimizer state.
   * \note Weight gradient tensors that are also optimiser tensors will only be
   * disabled if both \c disableGradAccumulationTensorStreams and
   * \c disableOptimizerStateTensorStreams are `true`.
   */
  bool disableOptimizerStateTensorStreams = false;

  /**
   * Setting to only build the Poplar graph but not compile not.
   *
   * If `false`, the backend will build the Poplar graph but not compile it
   * into an Engine.  In this case, no execution can be performed,
   * and nothing can be transferred to the device. API calls which retrieve
   * information from the graph building stage, such as tile mapping
   * introspection, can still be used.
   */
  bool compileEngine = true;

  /**
   * Specify an optimization for an inference session to have constant weights.
   *
   * Set this option to `false` in order to change the
   * weights with a call to Session::resetHostWeights() after the session has
   * been prepared. This option has no effect on a training session.
   *
   * Default: `true`.
   */
  bool constantWeights = true;

  /**
   * Enable Poplar executable caching.
   * The file is saved to the location defined with <tt>cachePath</tt>.
   * The file will be in
   * the <a href="https://docs.graphcore.ai/projects/popef/"> PopEF</a> format.
   * This means that it can be used to run inference using the <a
   * href="https://developer.nvidia.com/nvidia-triton-inference-server">Triton
   * Inference Server</a> because Graphcore provides a backend to it. See the <a
   * href="https://docs.graphcore.ai/projects/poplar-triton-backend/"> Poplar
   * Triton Backend user guide</a> for more information.
   *
   * Default: `false` (not enabled).
   */
  bool enableEngineCaching = false;

  /**
   * Enable variable caching.
   *
   * This means that the caching process will save variables as additional <a
   * href="https://docs.graphcore.ai/projects/popef/">PopEF</a> blobs to the
   * file location defined with <tt>cachePath</tt>. If PopART will require data
   * for variables (during cache reading process), they will be automatically
   * read from the cache file.
   *
   * Note, turning this off allows a PopART Session to optimise the host memory
   * it consumes during model runtime. Specifically, weightsToHost() can write
   * directly to the IR tensor data buffers. If the option were on, this would
   * not be safe and the session would have to create separate buffers to write
   * the fetched data to.
   *
   * Default: `true` (enabled).
   */
  bool enableVariablesCaching = true;

  /// Folder to save the \c poplar::Executable to.
  std::string cachePath = "session_cache";

  /**
   * Enable that exceptions are thrown when floating point errors occur.
   *
   * Default: `false` (not enabled).
   */
  bool enableFloatingPointChecks = false;

  /**
   * Enable stochastic rounding.
   *
   * PopART will set the Poplar engine option
   * `target.deterministicWorkers` to `true` if this option is set and to
   * `false` if it is not set. Adding a value for "target.deterministicWorkers"
   * to SessionOptions::engineOptions overrides this behaviour.
   *
   * Default: `false` (not enabled).
   */
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

  /**
   * Enable explicit recomputation.
   *
   * Default: `false` (not enabled).
   */
  bool explicitRecomputation = false;

  /**
   * Enable explicit pipelining.
   * Determined from values for `enablePipelining`, `useHostCopyOpsfault` and
   * `enableExplicitMainLoops`.
   */
  bool explicitPipeliningEnabled() const {
    return enablePipelining && useHostCopyOps && enableExplicitMainLoops;
  }

  /**
   * Enable implicit pipelining.
   * Determined from values for `enablePipelining`, `useHostCopyOpsfault` and
   * `enableExplicitMainLoops`.
   */
  bool implicitPipeliningEnabled() const {
    return enablePipelining && !(useHostCopyOps && enableExplicitMainLoops);
  }

  /**
   * Enable explicit representations in the IR (code paths).
   * Enabled if `true`, otherwise not.
   */
  void enableExplicitIR(bool enable) {
    useHostCopyOps          = enable;
    enableExplicitMainLoops = enable;
    explicitRecomputation   = enable;
  }

  /**
   * A wrapper class for the SessionOptions::numIOTiles option that permits any
   * int value and has an 'unassigned' state.
   */
  class NumIOTiles {
  public:
    /// Constructor.
    NumIOTiles();
    /**
     * Constructor.
     * \param numIOTiles The number of IPU tiles dedicated to IO.
     */
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
  // TODO: Remove with T19212
  bool delayVarUpdates = true;

  // When `shouldDelayVarUpdates` is `true`, the other ops in the proximity of
  // the delayed var updates may inherit the -inf schedule priority used to
  // delay the var updates. This is undesirable for some ops that consume
  // gradients, as it is preferred to consume (and thus be able to recycle the
  // memory of) those gradients as soon as possible. Two examples are
  // HistogramOp ops when doing automatic loss scaling, and the AccumulateOp ops
  // that accumulate the gradients when doing gradient accumulation.
  //
  // If `true` and if `shouldDelayVarUpdates` is `true`, this option will cause
  // the schedule priority of the above described ops to be re-overridden to
  // +inf. TODO: Remove with T19212.
  bool scheduleNonWeightUpdateGradientConsumersEarly = false;

  // TODO: Remove with T19212
  bool shouldDelayVarUpdates() const;

  // clang-off
  /**
   * Enable the global `fullyConnectedPass` option for matmuls.
   * \sa poplin::matMul(poplar::Graph, poplar::Tensor, poplar::Tensor,
   * poplar::program::Sequence, poplar::Type, poplar::DebugContext,
   * poplar::OptionFlags, matmul::PlanningCache).
   */
  // clang-on
  bool enableFullyConnectedPass = true;

  /// Enable/disable the serializing of matmuls.
  bool enableSerializedMatmuls = true;

  // For `partialsTypeMatMuls`, possible values are defined by
  // `fromString` in op/matmul.cpp. As of last check, those are:
  // "float", "half" in any letter case.

  /**
   * Set the partials type globally for matmuls. Can be overridden individually
   * with Builder.setPartialsType(). Valid values are `"float"` and `"half"`.
   * By default, this is not set, so no global partials type is imposed.
   */
  std::string partialsTypeMatMuls;

  /**
   * If `true`, computes the mean first and subtracts the activations
   * from it before computing the variance. The implementation with
   * this flag set to `true` is slower than when set to `false`.
   * The stable version requires the first order moment to be
   * estimated and applied to the sample set before the second
   * order central moment is calculated.
   */
  bool enableStableNorm = false;

  /// Poplar engine options.
  std::map<std::string, std::string> engineOptions;

  /// Poplar convolution options.
  std::map<std::string, std::string> convolutionOptions;

  /// Poplar LSTM options.
  std::map<std::string, std::string> lstmOptions;

  /// Poplar matmul options.
  std::map<std::string, std::string> matmulOptions;

  /// Poplar reporting options.
  std::map<std::string, std::string> reportOptions;

  /// GCL options.
  std::map<std::string, std::string> gclOptions;

  struct ExperimentalSettings {
    /**
     * Custom transform applier settings. Enable to insert custom transform
     * sequence at predefined checkpoint. Multiple checkpoint names and
     * transform names can be passed for different model configurations.
     *
     * The predefined checkpoint names are:
     * FWD0: Initial IR immediately after lowering from ONNX to the IR.
     *
     * FWD1: After the pre-alias patterns have been applied to FWD0.
     *
     * BWD0: After growing the backward pass (including the optimiser step).
     * Note this happens before optimiser decomposition, so the optimiser will
     * appear as a single special op rather than the many ops that implement it.
     *
     * PREALIAS: After pre-alias transforms have been applied to BWD0.
     *
     * MAINLOOPS: After the MainLoops transform has been applied. This transform
     * adds explicit loop ops to the IR for device iterations (batches per step)
     * and gradient accumulation.
     *
     * FINAL: The final IR after preparation.
     *
     * The transform names are defined by PopART and users.
     *
     * For example to execute 'Transform A' and 'Transform B' at 'Fwd0'
     * checkpoint and exectue 'Transform C' at 'Fwd1' checkpoint:
     *
     * {
     *  "Fwd0": [
     *    "Transform A",
     *    "Transform B"
     *  ],
     *  "Fwd1": [
     *    "Transform C"
     *  ]
     * }
     *
     * \note This setting is experimental for inference and may change.
     */
    std::map<std::string, std::vector<std::string>>
        customTransformApplierSettings;
  };

  /// Configuration setting for custom transform applier.
  ExperimentalSettings experimentalSettings;

  /**
   * List of codelet files (with file extension) to be added to the Poplar
   * graph. See the Poplar documentation for poplar::Graph for more information.
   */
  std::vector<std::string> customCodelets;

  /**
   * List of model named buffers that can be updated with call to
   * copyNamedBuffersToDevice(). This allows to update just a subset of model
   * weights instead of all or them as it happens with
   * copyWeightsToDevice() call.
   */
  std::vector<TensorId> updatableNamedBuffers;

  /**
   * Compile flags for the custom codelets. For example `-g` to generate debug
   * info. See the Poplar documentation for poplar::Engine for more information.
   */
  std::string customCodeletCompileFlags;

  /**
   * The maximum allowed time (in seconds) that can be spent searching for a
   * good graph schedule before a solution must be returned.
   */
  double timeLimitScheduler = 1e9;

  /**
   * The maximum number of improving steps allowed by the scheduling algorithm
   * before a solution must be returned.
   */
  int64_t swapLimitScheduler = static_cast<int64_t>(1e9);

  /**
   * The directory to serialize Poprithms graphs to.
   *
   * PopART uses Poprithms for scheduling PopART graphs. The Poprithms graphs
   * created for scheduling can be optionally serialised (written to file). If
   * `serializedPoprithmsShiftGraphsDir` is empty, then the graphs will not be
   * serialised. The names of serialization files will be
   * `poprithms_shift_graph_i.json` for the lowest non-existing values of `i`.
   * The directory must already exist, PopART will not create it.
   */
  std::string serializedPoprithmsShiftGraphsDir{};

  // clang-off
  /**
   * Specify which method is used to control how ops are scheduled.
   *
   * The initial scheduling is done with Kahn's algorithm. When several ops are
   * free to be scheduled, this controls which method is used.
   *
   * Options are described in the [Poprithms KahnTieBreaker
   * enum](https://github.com/graphcore/poprithms/blob/sdk-release-2.4/poprithms/poprithms/include/poprithms/schedule/shift/kahndecider.hpp).
   */
  // clang-on
  std::string kahnTieBreaker = "greedy";

  /**
   * Specify the transitive closure optimization threshold.
   *
   * The transitive closure optimization pass can significantly accelerate the
   * scheduler. It does not, in general, affect the final schedule returned. It
   * is run between initialization with Kahn's algorithms and the shifting
   * swaps. The transitive closure optimization pass is O(nOps^2) and so should
   * not be used for extremely large graphs. If a graph is above this
   * threshold, the transitive closure optimization pass is not run.
   */
  size_t transitiveClosureOptimizationThreshold{100000};

  /**
   * Enable replacement of single sums of partial gradients with a tree of
   * additions.
   * This can reduce max liveness at the cost of extra cycles. A typical
   * use case for this would be if a large weight tensor is used as an
   * input to many operations.
   *
   * Default: `false` (not enabled).
   */
  bool decomposeGradSum = false;

  /// Control the behavior of different collective operations.
  ReplicatedCollectivesSettings replicatedCollectivesSettings;

  /**
   * Enable training with Poplar replicated graphs across multiple PopART
   * instances.
   *
   * Default: `false` (not enabled).
   */
  bool enableDistributedReplicatedGraphs = false;

  /**
   * The total number of replicas in a multi-instance, replicated-graph training
   * session (this should be left as the default value (1) if distributed
   * replicated graphs are disabled). This value includes local replication.
   */
  int64_t globalReplicationFactor = 1;

  /// The first replica index that this PopART instance is running.
  int64_t globalReplicaOffset = 0;

  /**
   * Get the global replication factor.
   *
   * \returns
   *       - If `enableDistributedReplicatedGraphs` is `true`, then return
   *             `globalReplicationFactor`.
   *       - If `enableReplicatedGraphs` is `true`, then return
   *             `replicatedGraphCount`.
   *       - otherwise return 1.
   */
  int64_t getGlobalReplicationFactor() const;

  /**
   * Get the gradient accumulation factor.
   *
   * Throws an error if gradient accumulation is not enabled
   * (`enableGradientAccumulation` is `false`) and the factor
   * (`accumulationFactor`) is set to >1.
   *
   * \returns The accumulation factor.
   */
  unsigned getAccumulationFactor() const;

  /**
   * Specify to group the streams from the host to the device at the beginning
   * of the schedule, and the streams from the device to the host at the end of
   * the schedule. This trades off memory usage for speed.
   *
   * When `true`, tensors will stay live for longer.
   * \note This setting has no effect when useHostCopyOps is enabled (`true`).
   *
   * Default: `false` (not enabled).
   */
  // See T62461
  bool groupHostSync = false;

  /**
   * Enable strict op version checks.
   *
   * Strict op version checks will throw an error if the exact version of an op
   * required for the model opset is not supported. Turning this check off
   * will cause PopART to fall back to the latest implementation of the op that
   * is supported.
   * \warning Turning off these checks may cause undefined behaviour.
   *
   * Default: `true` (enabled).
   */
  bool strictOpVersions = true;

  /**
   * Enable running Opx checks to verify that IR tensor aliasing information
   * corresponds to the lowered Poplar tensor aliasing.
   *
   * Default: `false` (not enabled).
   */
  bool opxAliasChecking = false;

  /**
   * Enable running Opx checks to verify that IR tensor modification information
   * corresponds to the lowered Poplar tensor modifications.
   *
   * Default: `false` (not enabled).
   */
  bool opxModifyChecking = false;

  /**
   * Enable use of IR graph operations for data and anchor streams.
   *
   * Default: `false` (not enabled).
   */
  bool useHostCopyOps = false;

  /**
   * Enable load and offload of device RNG state from host.
   *
   * Default: `false` (not enabled).
   */
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

  /**
   * Override tensor location for specific tensors by setting tensor locations
   * for specific tensor ID values.
   */
  std::map<TensorId, TensorLocation> tensorLocationSettingsOverride;

  /**
   * Settings to enable and configure the automatic loss scaling behaviour when
   * training.
   *
   * \note Automatic loss scaling is in preview. It is well tested
   * and enabled in some of our example applications, but may not behave
   * as expected in all models. Recommendation: if your model with
   * automatic loss scaling enabled does not converge or triggers a
   * compilation error, then you will need to set the loss scale manually.
   */
  AutomaticLossScalingSettings automaticLossScalingSettings;

  /// Settings for developers to configure testing and benchmarking.
  DeveloperSettings developerSettings;

  /**
   * Enable casting to supported data types.
   * If enabled (`true`), casts any tensor of unsupported data types to
   * supported data types when lowering to Poplar. Currently, this implies
   * casting:
   *    - INT64 -> INT32
   *    - UINT64 -> UINT32
   * The cast will throw an error for incompatible data types and
   * over/underflows, and will warn about narrowing casts.
   *
   * Default: `true` (enabled).
   */
  bool enableSupportedDataTypeCasting = true;

  /**
   * Enable explicit main loop transformation, and disable implicit training
   * loops.
   *
   * \note This will be deprecated and enabled by default.
   */
  bool enableExplicitMainLoops = false;

  /**
   * Enable fast math mode for group norms.
   *
   * Group norms have a fast math mode which changes the implementation to run
   * faster on IPU but as a consequence is incompatible with other
   * implementations (so for running trained weights on host).
   * The default (`false`) is to use the correct, but slightly slower mode.
   */
  bool groupNormStridedChannelGrouping = false;

  // Get the buffering depth for a TensorId. For tensors that are rearranged on
  // host this is always 1, and bufferingDepthMap[id] shouldn't exist.
  // Otherwise returns bufferingDepthMap[id] if it exists, and
  // defaultBufferingDepth if it doesn't.
  //
  // **Not part of public API**
  unsigned getBufferingDepth(const TensorId &id, bool rearrangedOnHost);

  /**
   * Callback function used to indicate PopART compilation progress.
   *
   * The function should not block. All calls to the callback function will be
   * made from the main thread so blocking in the callback will block
   * compilation from progressing.
   *
   * If this logger is not set then compilation progress will be printed on the
   * info channel.
   *
   * \param int The progress value.
   * \param int The maximum value for the progress.
   */
  std::function<void(int, int)> compilationProgressLogger;

  /// Total progress ticks until compilation complete.
  int compilationProgressTotal = 100;

  /// Returns `true` if auto-recomputation is enabled, `false` otherwise.
  bool autoRecomputationEnabled() const;

  /**
   * Enable merging remote and host IO operations to facilitate IO overlap.
   * `true` to enable, otherwise `false`.
   *
   * Default=`true`.
   */
  bool enableMergeExchange = true;

  /**
   * Ensure that the loss scale tensor is fp32 and that this is combined
   * with fp16 activations as late as possible to produce the first fp16
   * activation gradients. This makes it possible to choose a loss scale value
   * greater than max(fp16). This is also recommended when automatic loss
   * scaling is enabled.
   * Only compatible with models that have an fp16 loss scale tensor.
   * `true` ensures that the loss scale tensor is fp32.
   *
   * Default: `false`.
   */
  bool ensureFp32LossScaleTensor = false;

  /**
   * Enable creation of an \c AliasModel object for each graph and run
   * the Poprithms ambiguity checker on it.
   * This throws an error if the graph has a potential inplacing ambiguity.
   *
   * See \c poprithms::memory::inplace::Graph::AmbiguityStatus for more info on
   * what constitutes an ambiguity.
   *
   * If set to `true`, \c AliasModel object is created for each graph and the
   * the Poprithms ambiguity checker is run on it.
   * No ambiguity checking is performed if this option is set to `false`
   * (default). However inplace fallbacks will occur if necessary.
   */
  bool enableInplaceAmbiguityChecking = false;

  // TODO T52152: Remove implicit pipelining
  /// \deprecated Create a custom program containing the forward pipeline only.
  bool createImplicitPipeliningFwdOnlyProgram = false;

  /**
   * If set to `true`, throw a Poplar error if any fused ops that consume a
   * log2 scale tensor receive a log2 scale tensor value not in the integer
   * range [-32, 32).
   *
   * If set to `false`, no error is thrown. However, note that this may lead to
   * undefined behaviour if the value of the log2 scale is outside the range.
   */
  bool throwIfLog2ScaleTensorNotInRange = true;

  /**
   * If set to `false`, disable constant folding on ops if any input have
   * multiple consumers.
   *
   * Default=`true`.
   */
  bool enableConstantFoldingOfMultipleConsumers = true;

  /**
   * Use loop candidate creator for constant if one exsits.
   *
   * Default=`false`.
   */
  bool useLoopCandidateCreator = false;

  /**
   * Stash all tensors when inference pipeline.
   *
   * Default=`false`.
   */
  bool stashAllTensorsInferencePipeline = false;

  /// Constructor for SessionOptions.
  SessionOptions() {
    // Automatically set `enableEngineCaching` and `cachePath` if the
    // environment variable `POPART_CACHE_DIR` or `POPXL_CACHE_DIR` is provided
    auto popartCachePathEnv = getPopartEnvVar("CACHE_DIR");
    auto popxlCachePathEnv  = getPopXLEnvVar("CACHE_DIR");

    if (popartCachePathEnv && popxlCachePathEnv &&
        (*popartCachePathEnv != *popxlCachePathEnv)) {
      logging::warn("Both POPART_CACHE_DIR ('{}') and POPXL_CACHE_DIR ('{}') "
                    "are set and differ from each other. The value of "
                    "POPART_CACHE_DIR will be ignored.",
                    *popartCachePathEnv,
                    *popxlCachePathEnv);
    }

    if (popartCachePathEnv) {
      enableEngineCaching = true;
      cachePath           = *popartCachePathEnv;
    }

    if (popxlCachePathEnv) {
      enableEngineCaching = true;
      cachePath           = *popxlCachePathEnv;
    }
  }

private:
  // Need to make sure that this is both invalid and obscure, so that customers
  // don't set this particular value accidentally.
  static const unsigned initialDefaultPrefetchBufferingDepthValue;
};

} // namespace popart

namespace std {
template <> struct hash<popart::SessionOptions> {
  std::size_t operator()(const popart::SessionOptions &so) const;
};
} // namespace std

#endif // POPART_WILLOW_INCLUDE_POPART_SESSIONOPTIONS_HPP_
