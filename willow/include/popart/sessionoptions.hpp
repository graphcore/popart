// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_OPTION_FLAGS_HPP
#define GUARD_OPTION_FLAGS_HPP

#include <iterator>
#include <map>
#include <set>
#include <string>

#include <popart/op.hpp>
#include <popart/tensorlocation.hpp>

namespace popart {

// Stages of Ir construction where .dot files can be written
enum class DotCheck {
  Fwd0 = 0, // after construction of the forward pass
  Fwd1,     // after running pre-aliasing patterns
  Bwd0,     // after backwards construction
  PreAlias, // after all transformations, patterns, except the aliasing
  Final,    // after running aliasing patterns (the final Ir)
  N         // the number of DotChecks, must appear as the final enum
};

std::string getDotCheckString(DotCheck);
DotCheck dotCheckFromString(const std::string &);

// If doing auto-recomputation, how should we decide which ops to recompute
// in the backwards pass?
enum class RecomputationType {
  None = 0, // No ops should be recomputed
  Standard, // Algorithm to pick checkpoint to try an minimize max liveness
  NormOnly, // Only Norm ops (+ non-linearities, if following) are recomputed
  Pipeline, // Recompute all forward pipeline stages
  N         // the number of RecomputationTypes, must appear as the final enum
};

enum class MergeVarUpdateType {
  None = 0,  // Do not merge VarUpdate Ops
  All,       // Merge all VarUpdate Ops into as few groups as possible.
             // This is a good choice when memory is not a constraint
  AutoLoose, // Merge into groups, attempting to not increase max-liveness in
             // the process, and not slicing individual Var Tensors to be
             // processed by different VarUpdateOps
  AutoTight, // Merge into groups, so that VarUpdateOps process Tensors of
             // exactly mergeVarUpdateMemThreshold in size
  N          // The number of MergeVarUpdateTypes, must appear as the final enum
};

enum class VirtualGraphMode {
  Off = 0, // virtual graphs are not enabled
  Manual,  // user must set the virtualGraph attribute on all ops and losses
  Auto,    // autoVirtualGraph transform is used
  ExecutionPhases, // virtual graphs are tied to execution phases
  N // The number of VirtualGraphModes, must appear as the final enum
};

enum class IrSerializationFormat {
  JSON // JSON format
};

enum class SyntheticDataMode {
  Off = 0,      // Use real data
  Zeros,        // Input tensors are initialised to all zeros
  RandomNormal, // Input tensors are initialised with distribution ~N(0,1)
  N             // The number of SyntheticDataModes, the final enum
};

enum class Instrumentation {
  Outer = 0, // Outer loop instrumentation, graph over all IPUs
  Inner,     // Inner loop instrumentation, graph per IPU
  N          // The number of Instrumentations, the final enum
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

  // The default tensor location for this tensor type.
  TensorLocation location = TensorLocation();

  // A minimum number of elements below which offloading won't be considered.
  int minElementsForOffChip = 2;

  // Only enable RTS for tensors with more than 8192 elements
  int minElementsForReplicatedTensorSharding = 8192;
};

// Setting that affects batch serialisation subgraph schedule before outlining.
// This setting is experimental and may change.
enum class BatchSerializationBatchSchedule {
  // Don't encourage any particular scheduling for ops within batch subgraphs
  // (leave it to the scheduler) but tell the scheduler to schedule subgraphs
  // in sequence.
  Scheduler = 0,
  // Encourage all ops within batch subgraphs to be scheduled identically and
  // for each subgraph to be scheduled in sequence (good for outlineability).
  Isomorphic,
  // OverlapOnIo tries to put the RemoteLoad for batch N+1 right after the
  // compute phase of batch N.
  OverlapOnIo,
  // OverlapOnCompute tries to put the RemoteLoad for batch N+1 right before
  // the compute phase of batch N.
  OverlapOnCompute
};

/**
 * A structure containing batch serialization settings.
 */
struct BatchSerializationSettings {
  BatchSerializationSettings() = default;
  BatchSerializationSettings(int factor_,
                             bool concatOnVirtualGraphChange_,
                             bool concatOnExecutionPhaseChange_,
                             bool concatOnPipelineStageChange_,
                             BatchSerializationBatchSchedule batchSchedule_ =
                                 BatchSerializationBatchSchedule::Isomorphic);

  BatchSerializationSettings &
  operator=(const BatchSerializationSettings &rhs) = default;

  int factor                        = 0;
  bool concatOnVirtualGraphChange   = true;
  bool concatOnExecutionPhaseChange = true;
  bool concatOnPipelineStageChange  = true;
  // This setting is experimental and may change.
  BatchSerializationBatchSchedule batchSchedule =
      BatchSerializationBatchSchedule::Isomorphic;
};

enum class ExecutionPhaseIOSchedule {
  // Preload tensors in previous phase for use in current phase
  Preload = 0,
  // Load tensors just before they are required
  OnDemand,
};

enum class ExecutionPhaseSchedule {
  // The steps for phased execution consists of:
  // 1. Copy to IO tiles if necessary
  // 2. Run collective operations if necessary
  // 3. Load optimizer state
  // 4. Update optimizer state
  // 5. Apply optimizer
  // 6. Store updated tensor if necessary

  // Process above steps for one weight at a time,
  // or as interleaved as the scheduler decides to minimize liveness
  // e.g.  (123456, 123456, 123456)
  Interleaving = 0,
  // Process above steps for all weights together, in a way that maximizes
  // overlap potential between compute and exchange
  // e.g. (333, 111, 222, 444, 555, 666)
  Batch
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

  // Number of ExecutionPhases for the whole model
  int phases = 0;

  // Number of overlapping stages
  // 1: Parallel streaming memory, default for 1 IPU / replica
  // 2: PingPong between 2 IPUs, default for >= 2 IPUs / replica
  int stages = 2;

  ExecutionPhaseIOSchedule weightIOSchedule = ExecutionPhaseIOSchedule::Preload;
  ExecutionPhaseIOSchedule activationIOSchedule =
      ExecutionPhaseIOSchedule::Preload;
  ExecutionPhaseIOSchedule optimizerStateIOSchedule =
      ExecutionPhaseIOSchedule::OnDemand;
  ExecutionPhaseIOSchedule accumulatorIOSchedule =
      ExecutionPhaseIOSchedule::Preload;

  ExecutionPhaseSchedule schedule = ExecutionPhaseSchedule::Interleaving;
};

/**
 * A structure containing user configuration options for the Session class
 */
struct SessionOptions {

  SessionOptions &operator=(const SessionOptions &rhs) = default;

  /// A directory for log traces to be written into
  std::string logDir;

  /// When to write '.dot' files during Ir construction
  std::set<DotCheck> dotChecks = {};

  /// The ops to write to the .dot file will be a continuous interval
  /// of the schedule, controlled by firstDotOp and finalDotOp. In particular,
  /// it will be [min(0, firstDotOp), max(N ops in Ir, finalDotOp))
  int firstDotOp = 0;
  int finalDotOp = 10000;

  /// Include the Op name in the .dot file (the Op type is always exported)
  bool dotOpNames = false;

  /// Export Poplar computation graph
  bool exportPoplarComputationGraph = false;

  /// Export Poplar vertex graph
  bool exportPoplarVertexGraph = false;

  bool separateCallOpPdfs = true;

  /// Controls caching of identical sections of the graph.
  bool enableOutlining = true;

  /// Controls whether the cost of copying of cached sections should be included
  /// in the outlining cost model.
  bool enableOutliningCopyCostPruning = true;

  /// The incremental value that a sub-graph requires, relative to its nested
  /// sub-graphs (if any), to be eligible for outlining. A high threshold
  /// results in fewer sub-graphs being outlined, a negative value results in
  /// all being outlined. The gross value of a sub-graph is the sum of its
  /// constituent Ops' getSubgraphValue() values. To disable outlining, it is
  /// better to set enableOutlining to false than to set this value to infinity.
  /// The default value of 1.0f results in all high Value operations such as
  /// convolution being cached, but standalone low Value operations such as Relu
  /// will not be.
  float outlineThreshold = 1.0f;

  /// Enable recomputation of operations in the graph in the backwards pass to
  /// reduce model size at the cost of computation cycles
  RecomputationType autoRecomputation = RecomputationType::None;

  /// Enable merging of VarUpdates into groups of VarUpdates, by flattening
  /// and concatenating Variable Tensors and Updating Tensors
  MergeVarUpdateType mergeVarUpdate = MergeVarUpdateType::None;

  /// The AutoLoose and AutoTight VarUpdate merging algorithm has a threshold on
  /// the total memory of Variable Tensors to merge for updating. Memory in
  /// bytes.
  int64_t mergeVarUpdateMemThreshold = 1000000;

  /// The AutoLoose VarUpudate merging algorithm has absolute threshold defined
  /// by min(mergeVarUpdateMemThreshold,
  ///        liveAtPeak - liveCurrently + looseThresholdAtPeak),
  /// where liveAtPeak is an estimate of the maximum live memory of the
  /// computation, and liveCurrently is an estimate of the live memory where the
  /// threshold is being used to determine whether to schedule or postpone a
  /// VarUpdate.
  int64_t looseThresholdAtPeak = 8000;

  /// Before anchor tensors are streamed from device to host, they are not
  /// necessarily arranged in memory as required when they are to be copied
  /// from host stream to host. This can be done on the device or on the host.
  /// Done on host by default to save memory, but often at the expense of
  /// cycles, especially for larger anchor tensors.
  bool rearrangeAnchorsOnHost = true;

  /// By default, we will use prefetching for input data streams. Poplar will
  /// speculative read data for a stream before is is required to allow the
  /// 'preparation' of the data to occur in parallel with compute
  bool enablePrefetchDatastreams = true;

  /// By default, we use the stable-softmax poplar function. This input tensor
  /// to softmax, _x_, is preprocessed by subtracting max(_x_) to each element
  /// before computing the exponentials, ensuring numerical stability. If you
  /// are sure the inputs to your softmax operations are small enough to not
  /// cause overflow when computing the exponential, you can enable the
  /// non-stable version instead for speed-up
  bool enableNonStableSoftmax = false;

  /// Enable replication of graphs
  bool enableReplicatedGraphs = false;

  /// Enable gradient accumulation
  bool enableGradientAccumulation = false;

  /// If enableReplicatedGraphs is true, replicatedGraphCount will set the
  /// number of model replications. E.g. if your model uses 1 IPU, a
  /// replicatedGraphCount of 2 will use 2 IPUs. If your model is
  /// pipelined across 4 IPUs, a replicatedGraphCount of 4 will use 16 IPUs
  /// total. Therefore the number of IPUs you request must be a multiple of
  /// replicatedGraphCount. If the training is done across multiple instances
  /// then the replicatedGraphCount is the number of replicas for this instance.
  int64_t replicatedGraphCount = 1;

  /// Specify the number of micro-batches to accumulate before applying the
  /// varUpdate.
  int64_t accumulationFactor = 1;

  /// This option allows you to place ops on virtual graphs to achieve model
  /// parallelism - either manually using model annotations, or automatically
  VirtualGraphMode virtualGraphMode = VirtualGraphMode::Off;

  /// Enable pipelining of virtual graphs
  bool enablePipelining = false;

  /// Use synthetic data i.e. disable data transfer to/from the host
  /// Set to 'Off' to use real data
  SyntheticDataMode syntheticDataMode = SyntheticDataMode::Off;

  /// Add instrumentation to your program to count the number of device cycles
  /// (a single tile, on a single IPU) that your main program takes to execute.
  /// Expect this to have a small detrimental impact on performance.
  bool instrumentWithHardwareCycleCounter            = false;
  std::set<Instrumentation> hardwareInstrumentations = {Instrumentation::Outer};

  /// If true, the weight gradient tensors are not saved off the device
  /// when devicex.weightsFromHost() is called. Note: this option is
  /// overridden if syntheticDataMode is not Off.
  bool disableGradAccumulationTensorStreams = false;

  /// when false, the backend will build the Poplar graph, but do not compile it
  /// into an Engine.  When this option is set, no execution can be performed,
  /// and nothing can be transferred to the device.  Functions which retrieve
  /// information from the graph building stage will be ok (tile mapping).
  bool compileEngine = true;

  // An optimization for an inference session to have constant weights, true by
  // default. Set this option to false if you are going to want to change the
  // weights with a call to resetHostWeights after the session has been
  // prepared. This option has no effect on a training session
  bool constantWeights = true;

  /// Enable poplar executable caching
  bool enableEngineCaching = false;

  /// Path to save the poplar::Executable to.
  std::string cachePath = "session_cache";

  // Enable exceptions when floating point errors occur.
  bool enableFloatingPointChecks = false;

  // Enable stochastic rounding
  bool enableStochasticRounding = false;

  // Configure execution phases
  ExecutionPhaseSettings executionPhaseSettings;

  // Enable explicit recomputation
  bool explicitRecomputation = false;

  // Number of IO tiles
  int numIOTiles = 0;

  // Enable zero-copy for subgraphs
  bool aliasZeroCopy = false;

  // Configure batch serialization
  BatchSerializationSettings batchSerializationSettings;

  // Delay var updates as much as possible
  // TODO: Remove with T19212
  bool delayVarUpdates = true;

  // Enable the global fullyConnectedPass option for matmuls
  bool enableFullyConnectedPass = true;

  /// Enable/disable the grouping of matmuls that are the same shape
  bool enableGroupedMatmuls = false;

  /// Enable/disable the serializing of matmuls.
  bool enableSerializedMatmuls = true;

  // Set the partials type globally for matmuls. Can be overriden individually
  // with `builder.setPartialsType()`. Possible values are defined by
  // `fromString` in op/matmul.cpp. As of last check, those are:
  // "float", "half" in any letter case.
  // By default, this is not set, so no global partials type is imposed.
  std::string partialsTypeMatMuls;

  /// If true, computes the mean first and subtracts the activations
  /// from it before computing the variance. The implementation with
  /// this flag set to true is slower than when set to false.
  /// The stable version requires the first order moment to be
  /// estimated and applied to the sample set before the second
  /// order central moment is calculated.
  bool enableStableNorm = false;

  /// Perform AllReduce operation on the host. Only useful for training session
  bool hostAllReduce = false;

  /// Perform weight update on the host. Only useful for training session
  bool hostWeightUpdate = false;

  /// Enable the use of poplar::RemoteBuffers for hostAllReduce operations
  bool hostAllReduceRemoteBuffer = false;

  /// Poplar engine options
  std::map<std::string, std::string> engineOptions;

  /// Poplar convolution options
  std::map<std::string, std::string> convolutionOptions;

  /// Poplar reporting options
  std::map<std::string, std::string> reportOptions;

  /// List of codelets (with filetype) to be added to the poplar graph. See the
  /// poplar documentation for more information.
  std::vector<std::string> customCodelets;

  /// Compile flags for the custom codelets. For example `-g` to generate debug
  /// info.
  std::string customCodeletCompileFlags;

  /// The maximum allowed time that can be spent searching for a good Graph
  /// schedule before a solution must be returned
  double timeLimitScheduler = 1e9;

  /// The maximum number of improving steps allowed by the scheduling algorithm
  /// before a solution must be returned
  int64_t swapLimitScheduler = static_cast<int64_t>(1e9);

  /// PopART uses Poprithms for scheduling PopART Graphs. The Poprithms Graphs
  /// created for scheduling can be optionally serialized (written to file). The
  /// string below specified the directory to serialize Poprithms Graphs to. If
  /// it is empty, then the Graphs will not be serialized. The names of
  /// serialization files will be poprithms_anneal_graph_`i'.json for the lowest
  /// non-existing `i's. The directory must already exist, PopART will not
  /// create it.
  std::string serializedPoprithmsAnnealGraphsDir{};

  /// The initial scheduling is done with Kahn's algorithm. When several Ops are
  /// free to be scheduled, this controls which method is used
  std::string kahnTieBreaker = "greedy";

  /// Replaces single sums of partial gradients with a tree of additions.
  /// This can reduce max liveness at the cost of extra cycles. A typical
  /// use case for this would be if a large weight tensor is used as an
  /// input to many operations
  bool decomposeGradSum = false;

  /// Enable training with Poplar replicated graphs across multiple PopART
  /// instances
  bool enableDistributedReplicatedGraphs = false;

  /// The total number of replicas in a multi instance replicated graph training
  /// session
  int64_t globalReplicationFactor = 1;

  /// The first replica index that this PopART instance is running
  int64_t globalReplicaOffset = 0;

  /// The IPU SystemType
  std::string ipuSystemType = "ipu1";

  /// Allows to group the streams from host at the beginning and the streams
  /// to host at the end, this trades off sum-liveness efficiency for cycle
  /// efficiency.
  bool groupHostSync = false;

  /// Strict op version checks will throw an error if the exact version of an op
  /// required for the models opset is not supported. Turning this check off
  /// will cause popart to fall back to the latest implementation of the op that
  /// is supported. Warning, turning off these checks may cause undefined
  /// behaviour.
  bool strictOpVersions = true;

  // Run Opx checks to verify IR tensor aliasing information
  // corresponds to lowered Poplar tensor aliasing
  bool opxAliasChecking = false;

  // Run Opx checks to verify IR tensor modification information
  // corresponds to lowered Poplar tensor modifications
  bool opxModifyChecking = false;

  // Tensor location settings for activation/gradient tensors.
  TensorLocationSettings activationTensorLocationSettings =
      TensorLocationSettings{TensorLocation(), 2, 8192};
  // Tensor location for weight tensors.
  TensorLocationSettings weightTensorLocationSettings =
      TensorLocationSettings{TensorLocation(), 2, 8192};
  // Tensor location for optimizer state tensors.
  TensorLocationSettings optimizerStateTensorLocationSettings =
      TensorLocationSettings{TensorLocation(), 2, 8192};
  // Tensor location for gradient accumulator tensors.
  TensorLocationSettings accumulatorTensorLocationSettings =
      TensorLocationSettings{TensorLocation(), 2, 8192};

  // Overriding tensor location for specific tensors.
  std::map<TensorId, TensorLocation> tensorLocationSettingsOverride;
};

} // namespace popart

namespace std {
template <> struct hash<popart::SessionOptions> {
  std::size_t operator()(const popart::SessionOptions &so) const;
};
} // namespace std

#endif
