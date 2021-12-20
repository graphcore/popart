
// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_WILLOWIR_HPP
#define GUARD_NEURALNET_WILLOWIR_HPP

#include <map>
#include <memory>
#include <set>

#include <popart/alias/aliasmodel.hpp>
#include <popart/bimap.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/names.hpp>
#include <popart/opidentifier.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/scheduler_requireoptimal.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensorindex.hpp>
#include <popart/transforms/pipeline.hpp>

namespace poprithms {
namespace logging {
class TimePartitionLogger;
}
} // namespace poprithms

namespace popart {

struct PTensorCmp;
enum class RequireOptimalSchedule; /*
  Yes = true,
  No = false
*/

// Ir Constructor inputs
class IrBundle {
public:
  IrBundle(const ONNX_NAMESPACE::ModelProto &modelProto,
           const InputShapeInfo &inputShapeInfo,
           const DataFlow &dataFlow,
           const TensorId &loss,
           const Optimizer *optimizer,
           DeviceInfo &deviceInfo,
           const SessionOptions &userOptions,
           const Patterns &patterns,
           const std::string sessionName = "");

  const ONNX_NAMESPACE::ModelProto &modelProto;
  const InputShapeInfo &inputShapeInfo;
  const DataFlow &dataFlow;
  const TensorId loss;
  const Optimizer *optimizer;
  DeviceInfo &deviceInfo;
  const SessionOptions &userOptions;
  const Patterns &patterns;
  const std::string sessionName;
};

/// Stores the shape and the numerical type of the remote buffer
class RemoteBufferInfo {
public:
  /**
   * Construct a new Remote Buffer Info object.
   *
   * \param info_ Shape and numerical type of the tensor stored in the remote
   *              buffer
   * \param repeats_ How many tensors of the same type and numerical shall the
   *                 remote buffer allocate space for
   */
  RemoteBufferInfo(TensorInfo info_, uint64_t repeats_)
      : info(info_), repeats(repeats_) {}
  RemoteBufferInfo() = default;
  TensorInfo info;
  uint64_t repeats;
};

using HashesMap = std::map<size_t, std::string>;

class Ir {

public:
  /**
   * \return An object used to track and summarize where wall clock time is
   *         spent in PopART compilation. This object is used to partition time
   *         into different components (scheduling, outlining, poplar Graph
   *         construction, etc.). It can be used as follows:
   *
   * <code>
   * void foo() {
   *     auto timer = timePartitionLogger().scopedStopwatch("In foo");
   *     if (cond0()){
   *        return;
   *     }
   *     bar();
   *     return;
   * }
   * </code>
   *
   * When the method timePartitionLoggerStr() (see below) is called, there will
   * be a line with "In foo" summarizing the time between between the
   * construction and destruction of \a timer, above. Something like:
   *
   * In foo          : 0.03 [s]    :     30 %
   * In bar          : 0.02 [s]    :     10 %
   * unaccounted     : 0.05 [s]    :     50 %
   * total           : 0.10 [s]    :    100 %.
   *
   * In the case where there are multiple timers which exist concurrently, only
   * the most recently constructed one will accumulate time. This means that the
   * most nested scope is the one which will accumulate time.
   *
   * For more information, see the poprithms SwitchingTimePartitionLogger class
   * */
  poprithms::logging::TimePartitionLogger &timePartitionLogger() const;

  std::string timePartitionLoggerStr() const;

  enum class ExecutionMode { Inference, Training };

  enum class SerialiseFormat { JSON };

  Ir();
  ~Ir();

  // NOTE: Ir owns Graph objects that have a reverse reference to the Ir, thus
  // moving the Ir would require updating those references. Thus, if you want to
  // implement movability for Ir, you will have to account for this.
  Ir(Ir &&)   = delete;
  Ir &operator=(Ir &&) = delete;

  Ir(const Ir &) = delete;
  Ir &operator=(const Ir &) = delete;

  // Unique id of each IR instance
  uint64_t getId() const { return id; }

  // Set the onnxModel.
  // A note on constant tensors: The outputs of ONNX Constant Operators
  // will always be treated as constants, so left unchanged if in training mode
  // Weights for training should always therefore rather appear in the ONNX
  // initializer list, and in the ONNX input list.
  void setOnnxModel(const ONNX_NAMESPACE::ModelProto &model);

  /**
   * Check if there's an ONNX model in the IR. This is true if the IR has been
   * created from an ONNX model or using the Builder.
   *
   * \return true If there is an onnx model, false otherwise.
   */
  bool hasOnnxModel() const { return onnxModel.get() != nullptr; }

  // Set the dataflow
  void setDataFlow(const DataFlow &df);

  // Set the user options
  void setUserOptions(const SessionOptions &flags);

  // Set the input shape information
  void setInputShapeInfo(const InputShapeInfo &info);
  const InputShapeInfo &getInputShapeInfo() const { return inputShapeInfo; }

  // Set the optimizer and add optimizer tensors
  // NOTE: could this be combined with updateOptimizerFromHost?
  void setOptimizer(const Optimizer &);

  void ensureOptimizerTensorCreated(const TensorId &optId,
                                    const TensorInfo &info);

  // Get the optimizer
  const Optimizer &getOptimizer() const { return *optimizer; }

  // Set the device info
  void setDeviceInfo(DeviceInfo &);

  const DeviceInfo *getDeviceInfo() const;

  // Set the optimization patterns
  void setPatterns(const Patterns &p);
  const Patterns &getPatterns() const { return patterns; }
  std::string getPatternLevelStr(const Patterns &p);
  bool isPatternsLevel(const Patterns &p, PatternsLevel level);

  // Remove from the IR any tensors which are unconnected, i.e.
  // the have no producers or consumers
  void removeIsolatedTensors(bool retainIoTensors);

  // Remove any Graphs that are not called by the main graph
  // (or by their called graphs; ad infinitum)
  void removeIsolatedGraphs();

  // Set which execution mode we are using
  void setExecutionMode(const ExecutionMode &mode);

  // Convenience methods to query the mode of the model.
  // Onnx refers to Inference as testing.
  bool isTraining() const { return executionMode == ExecutionMode::Training; }
  bool isTesting() const { return executionMode == ExecutionMode::Inference; }

  // Log the IR in a human readable format.
  void logIr() const;

  void compareWithSavedHash(const HashesMap &cacheEntries);

  // Prepare the IR based on the IrBundle configuration.
  // If engine caching is enabled then the IR hash which is
  // based on the IrBundle and the forward graph will be
  // compared to a saved file. If the hash matches then
  // the rest of the Ir preparation will be skipped.
  void prepare(const IrBundle &, const HashesMap &cacheEntries = {});

  // Prepare the IR based for a "IR model"
  // If engine caching is enabled then the IR hash which is
  // based on the IR graph will be compared to a saved file.
  // If the hash matches then the rest of the Ir preparation
  // will be skipped.
  void prepareCache(const HashesMap &cacheEntries);

  // Called once the IR 'prepare' is complete to finalize DebugInfo for each Op
  void finalizeOpDebugInfo();

  bool isPrepared() const { return isPrepared_; }
  bool hashMatched() const { return hashMatched_; }

  void updateOptimizer(const Optimizer &);

  // take training steps
  ONNX_NAMESPACE::ModelProto step(int n);

  // Registers an extra tensor to be added to the  model when calling
  // addAdditionalModelProtoTensors (such as additional optimizer state).
  // Will register the tensor regardless of whether the Ir has an Onnx model set
  // or not, but addAdditionalModelProtoTensors, which actually adds them to the
  // model proto, will throw if not.
  void addAdditionalModelProtoTensor(const TensorId &);
  void addAdditionalModelProtoTensor(Tensor *);

  // Adds an initializer to the Onnx model for all registered "additional model
  // proto tensors".
  //
  // \throws error if there are tensors to add, but the Ir has no Onnx model to
  // add them in.
  void addAdditionalModelProtoTensors();

  bool additionalModelProtoTensorsHaveBeenAdded() const {
    return additionalModelProtoTensorsAdded;
  }

  const std::set<Tensor *, PTensorCmp> &getAdditionalModelProtoTensors() const {
    return additionalModelProtoTensors;
  }

  std::set<Tensor *, PTensorCmp> &getAdditionalModelProtoTensors() {
    return additionalModelProtoTensors;
  }

  // if the tensor provides data that is returned to the host
  bool isAnchored(const TensorId &) const;

  // if the tensor was requested by the user as an anchor
  bool isRootAnchor(const TensorId &) const;

  // return all current anchors
  std::set<TensorId> getAnchors() const;

  // return all current root anchors
  std::set<TensorId> getRootAnchors() const;

  // remap anchor tensors (if the tensor to be copied to the host changes)
  void remapAnchor(const TensorId &from, const TensorId &to);
  const BiMap<TensorId, TensorId> &getAnchorRemap() const;

  bool streamingIsDisabledForTensor(const Tensor *) const;
  bool streamingIsDisabledForTensor(const TensorId &) const;
  bool storingIsDisabledForTensor(const Tensor *) const;
  bool storingIsDisabledForTensor(const TensorId &) const;
  void append(std::stringstream &) const;

  // Serialise the ir into the stream based on the format. In circumstances
  // where the scheduler may fail, setting `useScheduler` to false will not use
  // the scheduler and just return the ops and graphs in the order they are
  // stored.
  void serialise(SerialiseFormat format,
                 std::stringstream &ss,
                 bool useScheduler = true) const;

  // The optimizer parameter tensors such as learning rate(s), momentum(s),
  // weight decay factor(s), loss scaling
  std::vector<Tensor *> optimizerTensors() const;

  // The optimizer state variables such as accumulation and step tensors
  std::vector<Tensor *> optimizerStateTensors() const;

  /**
   * The original input tensor ID (used to identify streams) and the tensors
   * produced by associated HostLoadOp.
   *
   */
  std::map<TensorId, std::vector<Tensor *>> getHostLoadTensors() const;

  /**
   * The original anchor tensor ID (used to identify streams) and the tensors
   * consumed by associated HostStoreOp.
   *
   */
  std::map<TensorId, std::vector<Tensor *>> getHostStoreTensors() const;

  // The input data tensors. label(s), image(s), etc. This does not include
  // optimizer stream tensors (they are not data)
  std::vector<Tensor *> dataStreamTensors() const;

  std::vector<Op *> opsOfType(const OperatorIdentifier &opid) const;
  bool isConsumedByOpOfType(TensorId tid, const OperatorIdentifier &opid);

  // Simple recursive depth first search
  std::vector<const Graph *> getGraphSchedule() const;
  std::vector<const Graph *> getGraphSchedule(GraphId root) const;

  // Essentially Kahn's algorithm (1962),
  // https://en.wikipedia.org/wiki/Topological_sorting
  // with additional constrains imposed through the input paramater.
  // Ops which are ready to be inserted have an insertion "priority",
  // set elsewhere.
  //
  // Returns the op schedule across all graphs.
  //
  // Parameters:
  //   `const OpsBeforeKey &`: Extra topological constraints.
  //   `RequireOptimalSchedule ros`:
  //         Whether the true optimal schedule is required, which could be very
  //         expensive to compute; or whether merely any valid topological
  //         traversal is required.
  //         Note, the schedule is cached, but there may still be a cache miss
  //         if the graph has changed, or if an optimal schedule is required but
  //         the cached one is not optimal.
  //
  // Returns:
  //   `std::vector<Op *>`: The ops in schedule order.
  std::vector<Op *> getOpSchedule(const OpsBeforeKey &,
                                  RequireOptimalSchedule ros) const;

  // Do all the Ops with all their dependencies form a DAG?
  bool isSchedulable(const OpsBeforeKey &) const;

  // Is the virtualGraphMode set to something other than VirtualGraphMode::Off.
  bool virtualGraphsEnabled() const;

  // Returns the options.syntheticDataMode flag
  SyntheticDataMode syntheticDataMode() const;

  // Returns false if options.useSyntheticDataMode is set to Off, otherwise
  // returns true
  bool useSyntheticData() const;

public:
  static bool usingEngineCache(const SessionOptions &, const DeviceInfo *);

  OpId getOpsCounter() const;
  OpId getAndIncrOpsCounter();
  TensorId getFinalLossId() const;
  // The OpId if the Op which produces the final loss tensor
  OpId getFinalLossOpId() const;
  // if check is in userOptions.dotChecks, then write the .dot file
  // in userOptions.logDir
  void dotCheckpoint(const Ir &ir, std::string check) const;

  /**
   * \returns const reference to the Onnx model.
   * \throws error if there is no Onnx model.
   */
  const ONNX_NAMESPACE::ModelProto &getModel() const;

  /**
   * \returns the id of every input tensor of the Onnx model. If there is no
   * Onnx model, returns empty.
   */
  std::vector<TensorId> getModelInputIds() const;

  /**
   * Set the Onnx TensorProto of the given tensor in the Onnx ModelProto.
   *
   * \throws error if this Ir has no Onnx model.
   */
  void setExternalTensorDataInfo(TensorId, const ONNX_NAMESPACE::TensorProto &);

  const SessionOptions &getSessionOptions() const { return userOptions; }
  SessionOptions &getSessionOptions() { return userOptions; }

  void setSessionName(const std::string name) { sessionName = name; }
  const std::string getSessionName() const { return sessionName; }

  std::vector<TensorId> getTensorIds(TensorType) const;
  Tensor *getTensor(const TensorId &) const;
  bool containsTensor(const TensorId &) const;
  std::vector<TensorId> getGraphInputIds() const;
  std::vector<TensorId> getGraphOutputIds() const;

  const Graph &getMainGraph() const;
  Graph &getMainGraph();

  // Returns all graphs in `graphs' in an unscheduled order
  std::vector<const Graph *> getAllGraphs() const;

  Graph &getGraph(const GraphId &) const;
  bool hasGraph(const GraphId &) const;

  Graph &createGraph(const GraphId &);
  void removeGraph(const GraphId &);

  std::map<OpId, std::unique_ptr<Op>> &getMainGraphOps();
  const std::map<OpId, std::unique_ptr<Op>> &getMainGraphOps() const;

  std::vector<Op *> getAllOps() const;

  /**
   * Returns the Op if it exists in any graph.
   * Throws an error if the Op could not be found.
   * \param opId The unique ID of the Op to find
   * \return     The Op pointer if found
   */
  Op *getOp(OpId opId) const;

  Tensors &getMainGraphTensors();
  const Tensors &getMainGraphTensors() const;

  // Accessors for the dataFlow
  const DataFlow &getDataFlow() const { return dataFlow; }

  // modify a Graph using a graph transformation
  // (public for unit testing only)
  void applyTransform(std::size_t transformId, Graph &graph);

  // run after creating the backwards pass, checks that
  // the user provided anchor tensors actually exist.
  // the user may have not used the correct gradient
  // tensor naming convention for example, this will
  // be caught here.
  void validateAnchors() const;

  ExecutionMode getExecutionMode() const;

  // Can the IR be used for inference.
  bool canInfer() const;

  // Can the IR be used for training.
  // This is true when there is a loss and an optimizer.
  bool canTrain() const;

  // returns true if constructBackwards has finished
  bool hasConstructedBackwards() const;

  // returns true if the various optimizer Decomposer patterns have finished
  bool hasDecomposedOptimizers() const;

  // returns true if there are initializers in the onnx model
  bool containsInitialisers() const;

  // returns true if tensor Id matches the name of any tensor the onnx
  // model's initialisers, otherwise false (including if the Ir has no onnx
  // model).
  bool tensorExistsInInitialisers(TensorId) const;

  // Convert the ONNX graph into the forwards pass of the IR
  // \throws error if the Ir has no ONNX model.
  void constructForwards();

  // Convert an ONNX graph into IR
  Graph &constructFromOnnxGraph(const ONNX_NAMESPACE::GraphProto &graph,
                                const Scope &scope);

  // Calls ConstExprUtil::foldConstants on the graph.
  void foldConstants(Graph &);

  // Construct the backwards pass of the IR by doing an autograd of the forward
  // pass
  void constructBackwards();

  // Register the input tensors of the ONNX graph.
  // For the ONNX input tensors, determines which are
  // Stream and which are Variable.
  // \throws error if the Ir has no Onnx model.
  void registerInputTensors();

  // For all vertices set the phase, and whether or not
  // there is a path to vertex in whose phase is BWD.
  void updateVertices();

  // Ensure that all virtual graph IDs are not set.
  // This can occur if the user has specified them but virtual graphs are turned
  // off globally
  void unsetAllVirtualGraphIds();

  // modify the Ir using all the registered pre-alias patterns
  void applyPreAliasPatterns(Graph &);

  void applyUpdateInplacePrioritiesForIpu();

  void applyInplacePattern(Graph &);

  // confirm that the names of the Const tensors
  // from the user (constTensors) are in the onnx Model
  // Can be run after the forward pass of Ir has been
  // constructed
  void confirmConstIds() const;

  // confirm that no tensors in input(), nodes() or preRunKnowledge()
  // use reserved naming conventions. A note on design: The decision
  // to NOT add an independent dimension to TensorId, used exclusively
  // by automatically named tensors, was that when printing TensorIds
  // there would still be the possibility of conflict (i.e. projection
  // to single string might result in conflict).
  void confirmNoReservedIds() const;

  // Set the final loss
  void setFinalLoss(const TensorId &loss);

  // Return the default opset version for a domain
  int getDefaultOpsetVersion(const std::string &domain) const;

  // Helper function to return the maximum virtual graph id (maximum number of
  // VGraphs), based on replication factor and number of IPUs. Equal to number
  // of IPUs // replicated graph factor if using replicated graphs, else equal
  // to number of IPUs.
  unsigned getMaxVirtualGraphId() const;

  // Return the opset version in use for a domain. Empty domain implies AiOnnx.
  // If the Ir has no Onnx model, returns `getDefaultOpsetVersion(domain)`.
  int getOpSetVersionFromModel(const std::string &domain) const;

  bool autoRecomputationEnabled() const {
    return userOptions.autoRecomputationEnabled();
  }

  bool hasReplicatedTensorSharding() const;

  // Checks if any inputs or anchors have overlapped IO enabled
  bool hasOverlappedIO() const;

  void setRequiresRandomSeed() { requiresRandomSeed_ = true; }
  bool getRequiresRandomSeed() const { return requiresRandomSeed_; }

  RandomReferenceId getAndIncrementRandomReferenceId();

  TensorId getOrSetRandomReferenceTensor(RandomReferenceId, TensorId);

  void mergeRandomReferenceIds(std::set<RandomReferenceId> &);

  void setRemoteBufferInfo(RemoteBufferId, RemoteBufferInfo);
  const RemoteBufferInfo getRemoteBufferInfo(RemoteBufferId) const;
  const std::map<RemoteBufferId, RemoteBufferInfo>
  getAllRemoteBufferInfos() const;

  void setExecutionPhasesReady() { executionPhasesReady = true; }
  bool getExecutionPhasesReady() { return executionPhasesReady; }

  PipelineStage getNumPipelineStages() const;
  PipelineInfo pipelineInfo() const;

  void setMainGraphPathFromLoss();

  /// Verifies that all tensors have valid \a TensorInfos
  void verifyTensorInfos() const;

  /**
   * Marks the Ir as "prepared". This means the Ir is now ready to be lowered.
   * Failing to do this before lowering the Ir will result in an error.
   */
  void setIsPrepared();

private:
  // Unique id of each IR instance
  const uint64_t id;

  void prepareImpl(const IrBundle &, const HashesMap &cacheEntries);

  // Accessors for the tensors in the top-level graph
  const Tensors &getTensors() const;
  Tensors &getTensors();

  // Get all tensors in all graphs
  std::map<TensorId, Tensor *> getAllTensors() const;

  // Get all tensorIds in all graphs
  std::set<TensorId> getAllTensorIds() const;

  // gradients are named automatically. To prevent them
  // getting names already taken by non-gradient tensors,
  // we check that a reserved pattern is not present.
  void confirmNonReservedId(const TensorId &tenId) const;

  void growGradientVarUpdateOp(const TensorId &varId,
                               AliasModel &mainGraphAliasModel);

  void growCopyVarUpdateOp(const TensorId &varId,
                           const TensorId &from,
                           AliasModel &mainGraphAliasModel);

  // Common code for the growGradient... and growCopy...
  void growVarUpdateOpInternal(OpId opId, AliasModel &mainGraphAliasModel);

  // Get the best virtual graph Id based on the graph Ids of producers of ts
  // to minimise graph<->graph communication
  OptionalVGraphId
  getVirtualGraphIdFromTensorProducers(std::vector<Tensor *> ts);

  // Verify the connectivity of the graph
  void verifyConnectivity() const;
  void verifyOpInputConnectivity(const Graph &graph) const;
  void verifyOpOutputConnectivity(const Graph &graph) const;
  void verifyTensorProducerConnectivity() const;
  void verifyTensorConsumerConnectivity() const;
  void verifyTensorIds() const;
  void verifyReplicatedTensorSharding() const;

  // Verifies that the virtual graph IDs (if used) are valid, on ops and losses
  // if specified
  void verifyVirtualGraphIds(bool postAutoVirtualGraphTransform) const;

  // Very that all virtual graph ids have not been initialised. (Used when
  // virtual graphs are disabled)
  void verifyVirualGraphIdsNotInitialized() const;

  void verifyVertexAttributesOnlyInMain() const;
  void verifyPipelineSettings() const;
  void verifyExecutionPhaseSettings() const;
  void verifyAliasZeroCopySettings() const;
  void verifyExplicitMainLoopsSettings() const;
  void verifyOverlapIOSettings() const;
  void verifyBatchSerializationSettings() const;
  void verifySubgraphs() const;
  void verifyRecomputeAttributes() const noexcept(false);
  void verifyDistributedReplicatedGraphSettings() const;

  void verifyExecutionContexts() const;
  void verifyPipelineStageAttributes() const;

  // Verify ConstExpr folding has removed input tensors
  // as expected
  void verifyConstExprFolding();
  bool isCandidateForConstExprFolding(const Tensor &tensor) const;
  std::set<Tensor *> getRootInputsToOp(Op *op);

public:
  PipelineStage getFinalLossPipelineStage() const;

private:
  DataFlow dataFlow;

  std::unique_ptr<poprithms::logging::TimePartitionLogger> timePartitionLogger_;

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> onnxModel;
  // Additional tensors that we want to add to the model proto when saving to a
  // .onnx file
  std::set<Tensor *, PTensorCmp> additionalModelProtoTensors;
  // A flag to record if the Ir's model proto has had any additional tensor
  // protos added.
  bool additionalModelProtoTensorsAdded = false;

  // learning rate, momentum, etc.
  // Optimizer needed to construct backwards pass:
  // if momentum the Ir is different
  std::unique_ptr<Optimizer> optimizer;
  DeviceInfo *deviceInfo = nullptr;
  SessionOptions userOptions;
  std::string sessionName;
  InputShapeInfo inputShapeInfo;

  // The set of patterns to apply after constructing
  // forwards and backwards passes
  Patterns patterns;

  // create an Op from a Node
  std::unique_ptr<Op> addOp(const Node &, const Scope &);

  std::map<GraphId, std::unique_ptr<Graph>> graphs;

  // total number of ops ever created
  OpId opsCounter{100};

  // Map of transform Id to enable flag
  std::map<std::size_t, bool> transformEnableMap;

  // Map of ops and their root inputs
  std::map<OpId, std::set<Tensor *>> opAndRootInputs;

  TensorId finalLossId;
  OpId finalLossOpId{-1000};
  bool constructedFinalLoss = false;
  bool constructedBackwards = false;
  bool decomposedOptimizers = false;

  ExecutionMode executionMode = ExecutionMode::Training;

  bool executionPhasesReady = false;
  bool isPrepared_          = false;
  bool hashMatched_         = false;

  bool requiresRandomSeed_ = false;

  // enable/disable a transform stage
  void enableTransform(std::size_t transformId, bool enable);

  RandomReferenceId randomReferenceId = 0;

  std::map<RandomReferenceId, TensorId> randomReferenceTensorMap;

  std::map<RemoteBufferId, RemoteBufferInfo> remoteBufferInfoMap;

  /**
   * Map between actual tensor that provides
   * the expected data and user-defined anchor tensor,
   * based on how the graph was transformed and the anchor return type
   */
  BiMap<TensorId, TensorId> anchorRemap;

  // Store a hash which can identify the Ir when deserializing
  // PopART state.
  nonstd::optional<size_t> hash_;

  size_t irBundleHash = 0;

public:
  // A "dummy" Op used to ensure that anchor tensors
  // will be copied out of sub-graphs, even if they
  // have no consumers external to the sub-graph.
  Op &getSubgraphAnchorPlaceholder();

  const decltype(graphs) &getGraphs() const { return graphs; }

  // Create a new intermediate tensor id with a unique name
  TensorId createIntermediateTensorId(const TensorId &base_id);

  // Create a new intermediate slice tensor id with a unique name
  TensorId createSliceTensorId(TensorId base_id, unsigned s, unsigned e);

  // Create a new intermediate batch slice tensor id with a unique name
  TensorId createConcatTensorId(TensorId base_id);

  GraphId createUniqueSubgraphId(GraphId base_id);

  // Accumulate outer fragment parallelizer bin constraints. Not really needed
  // for functionality but without these some models take a lot longer to
  // schedule.
  std::vector<std::vector<Op *>>
  getAccumulateOuterFragmentBinConstraints(const Graph &graph) const;

  size_t getHash() const;
  void computeHash();
  size_t getIrBundleHash() const;
  void setIrBundleHash(size_t);

  /**
   * Clone a graph.
   *
   * .. warning::
   *
   *    Does not support cloning of the main graph.
   *
   * The OpIds and TensorIds will differ between the original and the cloned
   * graph. Hence a map between the old OpId and cloned OpId will be returned.
   * The new graph can be obtained by ir.getGraph(newGraphId);
   *
   * \param originalGraphId The id of the graph to clone
   * \param newGraphId      The id of the cloned graph
   * \return A map between the OpIds in the original and new graphs
   * */
  std::map<OpId, OpId> cloneGraph(GraphId originalGraphId, GraphId newGraphId);

  // modify the Ir using with pattern matching
  // Returns true if a change to the Ir was made.
  bool applyPreAliasPattern(const PreAliasPattern *, Graph &);

private:
  uint64_t intermediate_tensor_counter{0};
  uint64_t subgraph_id_counter{0};
};

} // namespace popart

namespace std {
template <> struct hash<popart::Ir> {
  std::size_t operator()(const popart::Ir &ir) const;
};

template <> struct hash<popart::IrBundle> {
  std::size_t operator()(const popart::IrBundle &irBundle) const;
};

} // namespace std

#endif
