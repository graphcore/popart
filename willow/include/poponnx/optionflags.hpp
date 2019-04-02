#ifndef GUARD_OPTION_FLAGS_HPP
#define GUARD_OPTION_FLAGS_HPP

#include <iterator>
#include <map>
#include <set>
#include <string>

namespace poponnx {

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

// If doing auto-recomputation, how should we decide which ops to recompute
// in the backwards pass?
enum class RecomputationType {
  None = 0, // No ops should be recomputed
  Standard, // Algorithm to pick checkpoint to try an minimize max liveness
  NormOnly, // Only Norm ops (+ non-linearities, if following) are recomputed
  N         // the number of RecomputationTypes, must appear as the final enum
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

  /// The ops to write to the .dot file will be a continous interval
  /// of the schedule, controlled by firstDotOp and finalDotOp. In particular,
  /// it will be [min(0, firstDotOp), max(N ops in Ir, finalDotOp))
  int firstDotOp = 0;
  int finalDotOp = 10000;

  /// Include the Op name in the .dot file (the Op type is always exported)
  bool dotOpNames = false;

  /// Include annotation of sub-graphs in the .dot file
  bool dotSubgraphAnnotation = false;

  /// Export Poplar computation graph
  bool exportPoplarComputationGraph = false;

  /// Export Poplar vertex graph
  bool exportPoplarVertexGraph = false;

  /// Controls caching of the convolution graphs. If set to false, then none of
  ///  the convolutions will be cached.
  bool enableConvolutionGraphCaching = true;

  /// Controls caching of identifical sections of the graph.
  bool enableOutlining = true;

  /// Enable recomputation of operations in the graph in the backwards pass to
  /// reduce model size at the cost of computation cycles
  RecomputationType autoRecomputation = RecomputationType::None;

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

  /// If enableReplicatedGraphs is true, replicatedGraphCount will set the
  /// number of replicated graphs - must be a factor of the number of IPU's
  /// (CHECK)
  int64_t replicatedGraphCount = 1;

  /// Enable transformation pass that attempts to automatically place ops on
  /// virtual graphs to achieve model parallelism.
  bool autoVirtualGraph = false;

  /// Use synthetic data i.e. disable data transfer to/from the host
  /// Set to 'true' to use synthetic data, 'false' to use real data
  bool ignoreData = false;

  /// when false, the backend will build the Poplar graph, but do not compile it
  /// into an Engine.  When this option is set, no execution can be performed,
  /// and nothing can be transferred to the device.  Functions which retrieve
  /// information from the graph building stage will be ok (tile mapping).
  bool compileEngine = true;

  // An optimization for an inference session to have constant weights, true by
  // default Set this option to false if you are going to want to change the
  // weights with a call to resetHostWeights after the session has been
  // prepared. This option has no effect on a training session
  bool constantWeights = true;

  /// Poplar engine options
  std::map<std::string, std::string> engineOptions;

  /// Poplar convolution options
  std::map<std::string, std::string> convolutionOptions;

  /// Poplar reporting options
  std::map<std::string, std::string> reportOptions;
};

} // namespace poponnx

#endif
