#ifndef GUARD_OPTION_FLAGS_HPP
#define GUARD_OPTION_FLAGS_HPP

#include <iterator>
#include <map>
#include <set>
#include <string>

namespace popart {

// Stages of Ir construction where .dot files can be written
enum class DotCheck {
  FWD0 = 0, // after construction of the forward pass
  FWD1,     // after running pre-aliasing patterns
  BWD0,     // after backwards construction
  PREALIAS, // after all transformations, patterns, except the aliasing
  FINAL,    // after running aliasing patterns (the final Ir)
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
  N          // The numbe of MergeVarUpdateTypes, must appear as the final enum
};

enum class VirtualGraphMode {
  Off = 0, // virtual graphs are not enabled
  Manual,  // user must set the virtualGraph attribute on all ops and losses
  Auto,    // autoVirtualGraph transform is used
  N        // The number of VirtualGraphModes, must appear as the final enum
};

enum class IrSerializationFormat {
  JSON // JSON format
};

std::string toString(VirtualGraphMode);
std::ostream &operator<<(std::ostream &, VirtualGraphMode);

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

  /// Controls caching of the convolution graphs. If set to false, then none of
  ///  the convolutions will be cached.
  bool enableConvolutionGraphCaching = true;

  /// Controls caching of identifical sections of the graph.
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

  /// By default, we use the stable-softmax poplar function. This input tensor
  /// to softmax, _x_, is preprocessed by subtracting max(_x_) to each element
  /// before computing the exponentials, ensuring numerical stability. If you
  /// are sure the inputs to your softmax operations are small enough to not
  /// cause overflow when computing the exponential, you can enable the
  /// non-stable version instead for speedup
  bool enableNonStableSoftmax = false;

  /// Enable placement of operations on individual IPUs by creating a 'virtual
  /// graph' for each IPU
  bool enableVirtualGraphs = false;

  /// Enable replication of graphs
  bool enableReplicatedGraphs = false;

  /// Enable gradient accumulation
  bool enableGradientAccumulation = false;

  /// If enableReplicatedGraphs is true, replicatedGraphCount will set the
  /// number of replicated graphs - must be a factor of the number of IPU's
  /// (CHECK)
  int64_t replicatedGraphCount = 1;

  /// Specify the number of micro-batches to accumulate before applying the
  /// varUpdate.
  int64_t accumulationFactor = 1;

  /// Enable transformation pass that attempts to automatically place ops on
  /// virtual graphs to achieve model parallelism.
  bool autoVirtualGraph = false;

  VirtualGraphMode virtualGraphMode = VirtualGraphMode::Off;

  /// Enable pipelining of virtual graphs
  bool enablePipelining = false;

  /// Use synthetic data i.e. disable data transfer to/from the host
  /// Set to 'true' to use synthetic data, 'false' to use real data
  bool ignoreData = false;

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

  // Enable the global fullyConnectedPass option for matmuls
  bool enableFullyConnectedPass = true;

  /// Enable/disable the grouping of matmuls that are the same shape
  bool enableGroupedMatmuls = false;

  /// Enable/disable the serializing of matmuls.
  bool enableSerializedMatmuls = true;

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
};

} // namespace popart

#endif
