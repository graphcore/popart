// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IRLOWERING_HPP
#define GUARD_NEURALNET_IRLOWERING_HPP

#include <gcl/CollectiveBalancedReorder.hpp>

#include <popart/vendored/optional.hpp>

#include <poplar/DataStream.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>

#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>

#include <popart/aliaszerocopy.hpp>
#include <popart/devicemanager.hpp>
#include <popart/popx/creatorx.hpp>
#include <popart/popx/enigma.hpp>
#include <popart/popx/inittensor.hpp>
#include <popart/popx/linearmapper.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/popprograms.hpp>
#include <popart/popx/poptensors.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/popx/virtualgraph.hpp>

#include <popart/subgraphcopyingstrategy.hpp>
#include <popart/subgraphpartitioner.hpp>

#include <memory>
#include <set>
#include <tuple>
#include <popart/names.hpp>
// MutableVoidData is defined in here:
#include <popart/stepio.hpp>
#include <popart/tensordata.hpp>

namespace popart {
namespace liveness {
class LivenessAnalyzer;
} // namespace liveness
namespace popx {

// TODO: Find common location to share between devicex and IrLowering
class devicex_memory_allocation_err : public popart::memory_allocation_err {

  const poplar::graph_memory_allocation_error exception;
  const poplar::OptionFlags reportOptions;

public:
  devicex_memory_allocation_err(const devicex_memory_allocation_err &rhs);

  devicex_memory_allocation_err(const poplar::graph_memory_allocation_error &e,
                                const poplar::OptionFlags &_reportOptions);

  std::unique_ptr<memory_allocation_err> clone() const;

  std::string getSummaryReport() const;

  std::string getProfilePath() const;
};

using PopStreamId = std::string;

class CollectiveBalancedHostRearrangement;
class Devicex;
class PopOpx;

enum class ToHostStreamType { NonAnchor, NonSumAnchor, SumAnchor };

poplar::Type popType(const TensorInfo &);
poplar::Type popType(DataType);

// Log compilation progress for the whole compilation.
// For consistency we try to stick to roughly the same distribution as TF:
// 0-3    % ir preparation
// 3-4    % high level passes
// 4-7    % preplanning
// 7-40   % graph construction
// 40-100 % Poplar compilation
class ProgressLogger {
public:
  ProgressLogger(const SessionOptions &);
  void compilationStart();
  void preplanningStart();
  void preplanningEnd();
  void creatingSequence(int task, int numTasks);
  void complete();
  // Called by Poplar
  void operator()(int progress, int total);

private:
  // Return the ratio progress / total
  // Return the current position given the current position given a
  // ratio (progress / total) between a start and end points.
  int current(int start, int end, int progress, int total);

private:
  std::function<void(int, int)> callback_;
  int progressTotal;
};

class IrLowering {
private:
  const Ir &_ir;
  std::unique_ptr<snap::Graph> pGraph{nullptr};

  std::vector<VirtualGraph> virtualGraphs;

  std::shared_ptr<DeviceInfo> deviceInfo;

  ProgressLogger progressLogger;

  std::map<PipelineStage, VGraphId> getPipelineToVGraphIdMap() const;

  std::map<std::pair<ExecutionContext, TaskId>, std::vector<Op *>>
      contextOpRegistry;
  std::map<TaskId, std::vector<Op *>> requiredRecomputes;

  // Collective balanced reordering information for replicated ops
  std::map<TensorId, std::shared_ptr<gcl::CollectiveBalancedReorder>>
      collectiveReorders;

  // Store input tensors based on how they are allocated
  std::set<TensorId> linearlyCreatedInputTensors;
  std::set<TensorId> efficientlyCreatedInputTensors;

  bool prepareGraphHasBeenCalled_;

  nonstd::optional<poplar::Executable> cachedExecutable;
  bool usingCachedExecutable_ = false;

  // Option to trace the opx execution using printTensor. This can be useful in
  // determining where an exception has occurred. It is enabled by setting
  // POPART_OPX_TRACE environment variable to "1"
  bool opxTrace = false;
  poplar::Tensor opxTraceTensor;

  // A set of tensor ids to print during graph execution.
  // This is an alternative to the PrintTensor op.
  // Tensors are specified using the environment variable POPART_PRINT_TENSORS.
  // example: export POPART_PRINT_TENSORS="MatMul:0 L1:0"
  std::unordered_set<TensorId> printTensorIds;

  // This keeps track of whether there the accumulateOuterFragment is empty
  // TODO T12001 a class which encapsulates framgments which has this attribute.
  bool outerLoopFragEmpty = true;

  // Helper class to determine where to move input/output copies.
  std::unique_ptr<liveness::SubgraphCopyingStrategy> subgraphCopyingStrat;

  // Helper class to analyze the global IR schedule and tensor liveness
  std::unique_ptr<liveness::LivenessAnalyzer> livenessAnalyzer;

  // Helper class to reuse tensors and call subgraphs by reference
  std::unique_ptr<liveness::AliasZeroCopy> aliasZeroCopy;

  // Helper class to interpret results of liveness analyzer.
  std::unique_ptr<liveness::SubgraphPartitioner> subgraphPartitioner;

  snap::Tensor rngStateTensor;

  // Non-const tensors used to keep track of batch count, modulo the return
  // period
  std::map<ReturnPeriod, poplar::Tensor> batchCountingTensors;
  std::map<ReturnPeriod, poplar::Tensor> batchCountCheckingTensors;

  // Map tensors evenly across all tiles
  LinearMapper linearMapper;

  poplar::Tensor randomSeedTensor;

  // T11630: Combine the inputStreams/outputStreams with the
  // fromHostStreams/toHostAnchorStreams streams?

  //  poplar::Streams for poplar::Tensors,
  //  1) from host to device;
  std::map<TensorId, poplar::DataStream> fromHostStreams;

  // and 2) from device to host;
  std::map<TensorId, poplar::DataStream> toHostAnchorStreams;
  std::map<TensorId, poplar::DataStream> toHostWeightStreams;

  // Remote buffers
  std::map<RemoteBufferId,
           std::pair<poplar::RemoteBuffer, nonstd::optional<poplar::Tensor>>>
      remoteBuffers;

  // Streams for doing allreduce on host side
  std::map<TensorId, poplar::RemoteBuffer> hostReduceRemoteBuffers;
  std::map<TensorId, poplar::DataStream> toHostGradientStreams;
  std::map<TensorId, poplar::DataStream> fromHostGradientStreams;
  std::map<TensorId, poplar::DataStream> fromHostWeightLoadStreams;
  std::vector<TensorId> hostReduceStreamIds;

  // The maximum number of inputs on any Op
  int maxOpInputs;

  void setMetadataFromIr();

  void verifyTaskOrder(const std::vector<TaskId> &taskOrder) const;

  // Task to create a poplar::Tensor from nothing, choosing
  // the correct create call (createWeights, addLinearly, etc)
  InitTensorPtrs getInitTensorCreators(Tensor *);

  // Task to create a poplar::Tensor with methods defined by InitTensorPtrs
  PriTask initTensorTask(InitTensorPtrs inits);

  static TaskId initTensorTaskId(TensorId);

  PriTask initRandomSeed();
  static TaskId initRandomSeedTaskId();

  PriTask rngStateFromHost();
  static TaskId rngStateFromHostTaskId();
  PriTask rngStateToHost();
  static TaskId rngStateToHostTaskId();
  PriTask initRngStateTensor();
  static TaskId initRngStateTensorTaskId();

  PriTask setInitTensorValTask(Tensor *);
  static TaskId setInitTensorValTaskId(TensorId);

  // Task to create a poplar::Stream to write to poplar::Tensor
  // C++ Note: if a lambda function which modifies `this' is created
  // it must be const w.r.t this, even if it not run
  PriTask streamFromHostTask(TensorId streamTensorId,
                             std::vector<Tensor *> tensors);
  static TaskId streamFromHostTaskId(TensorId);

  // Task to append a Copy from poplar::Stream to poplar::Tensor
  PriTask fromHostTask(Tensor *tensor, poplar::program::Sequence &streamSq);

  static TaskId fromHostTaskId(TensorId);

  // Task to create a poplar::Stream to write from poplar::Tensor to host
  PriTask streamToHostTask(TensorId streamTensorId,
                           std::vector<Tensor *> tensors,
                           bool isAnchorStream);
  static TaskId streamToHostTaskId(TensorId, bool isAnchorStream);

  poplar::program::Sequence &getAnchorReturnFragment(Tensor *tensor);

  // Task to append a Copy to poplar::Stream from poplar::Tensor
  PriTask toHostTask(Tensor *tensor,
                     poplar::program::Sequence &,
                     ToHostStreamType) const;
  static TaskId toHostTaskId(TensorId, bool isAnchorStream);

  // Task to create an accumulator and scaleAddto to a poplar::Tensor to be
  // Copied on the final batch per step
  PriTask anchorReturnTypeSumTask(Tensor *tensor,
                                  poplar::program::Sequence &sq);
  static TaskId anchorSumTaskId(const TensorId &);

  // Task to create poplar::Tensors from nothing, specifically for
  // use in keeping track of the batch count
  PriTask initBatchCounterTensorsTask(poplar::program::Sequence &sq);
  static TaskId initBatchCounterTensorsTaskId();

  // Task to add a program to increment and check the batch count
  PriTask updateBatchCountTask(poplar::program::Sequence &sq);
  static TaskId updateBatchCountTaskId();

  // Task to append a Copy to poplar::Stream from poplar::Tensor every
  // N batches
  PriTask toHostEveryNBatchesTask(Tensor *tensor,
                                  ReturnPeriod N,
                                  poplar::program::Sequence &);

  PriTask opTask(Op *, double priority, TaskId prevOpTaskId);
  void opTaskFunc(TaskId taskId, Op *, SequenceMap &seqs);
  void pipelinedOpTaskFunc(TaskId taskId, Op *, SequenceMap &seqs);
  void growOpx(PopOpx *, SequenceMap::SequenceInterval seqInterval);

  static TaskId opTaskId(Op *);

  void addOpTasks(PriTasks &);

  static TaskId pipelinedCopyTaskId(Op *op);

  void addPipelinedCopyTasks(PriTasks &);

  PriTask pipelinedCopyTask(Op *, TaskId prevTaskId);

  bool doRearrangeOnHost(Tensor *tensor) const;

  // Determine stream properties.
  poplar::ReplicatedStreamMode getReplicatedStreamMode(Tensor *tensor) const;
  unsigned getBufferingDepth(Tensor *tensor) const;

  void initPoplarGraph();

  template <typename T>
  void setInitVal(Tensor *tensor, DataType dstType = DataType::UNDEFINED);
  void setInitValHalf(Tensor *tensor);

  void setFloatingPointBehaviour(snap::Graph &graph);
  void setStochasticRoundingBehaviour(snap::Graph &graph);

  using ConvPlanParams   = std::tuple<const poplar::Target *,
                                    const poplin::ConvParams,
                                    const poplar::OptionFlags *>;
  using MatMulPlanParams = std::tuple<const poplar::Target *,
                                      const poplin::MatMulParams,
                                      const poplar::OptionFlags *>;
  void prePlanConvolutions();
  void prePlanMatMuls();

  std::vector<std::string> cycleCountIds;
  PopTensors tensors_;

public:
  IrLowering(const Ir &,
             std::shared_ptr<DeviceInfo> deviceInfo,
             bool prepareGraphHasBeenCalled = false);
  const Ir &ir() const { return _ir; }

  PopPrograms progs;

  poplar::OptionFlags pooling_options;
  poplar::OptionFlags lstmOptions;
  poplar::OptionFlags gclOptions;
  poplar::OptionFlags engineOptions;
  poplar::OptionFlags reportOptions;

  void setDevicex(Devicex *d) { dv_p = d; }
  // Used for opx creation
  Devicex *dv_p = nullptr;

  // Return stored input tensors based on how they are allocated
  std::set<TensorId> getLinearlyCreatedInputTensors() const;
  void setLinearlyCreatedInputTensors(const std::set<TensorId> &s) {
    linearlyCreatedInputTensors = s;
  }
  void addLinearlyCreatedInputTensors(TensorId id) {
    linearlyCreatedInputTensors.insert(id);
  }

  std::set<TensorId> getEfficientlyCreatedInputTensors() const;
  void setEfficientlyCreatedInputTensors(const std::set<TensorId> &s) {
    efficientlyCreatedInputTensors = s;
  }
  void addEfficientlyCreatedInputTensors(TensorId id) {
    efficientlyCreatedInputTensors.insert(id);
  }

  bool tryInitTensorByPostIRAliasing(TensorId dstId,
                                     const ViewChangers &viewChangers);

  static std::string cycleCountStreamId(std::string id);
  const std::vector<std::string> &getCycleCountIds() const {
    return cycleCountIds;
  }
  void setCycleCountIds(const std::vector<std::string> &ids) {
    cycleCountIds = ids;
  }

  const PopTensors &tensors() const { return tensors_; }

  PopTensors &tensors() { return tensors_; }

  void instrumentWithHardwareCycleCounter(poplar::program::Sequence &,
                                          int64_t tileId = 0,
                                          std::string id = "");

  snap::Graph &graph() {
    if (pGraph == nullptr) {
      throw error(
          "snap::Graph is null when the lowering state is deserialized");
    }
    return *pGraph;
  }
  const snap::Graph &graph() const {
    if (pGraph == nullptr) {
      throw error(
          "snap::Graph is null when the lowering state is deserialized");
    }
    return *pGraph;
  }

  // Prepares the graph ready for poplar compilation
  void prepareGraph();

  // Load a poplar::Executable from a stream and set
  // `this->cachedExecutable'.
  void loadPoplarExecutable(std::istream &in);

  // Either return the executable in cachedExecutable
  // or compile `rootGraph' and try to save the generated executable before
  // returning it. After calling `getExecutable', `cachedExecutable' will always
  // be set to `nonstd::nullopt'.
  poplar::Executable getExecutable();

  std::string getSerializedGraph() const;

  // Return virtual graph mapping to IPU virtualGraphIndex,
  // ioTileGraph selects between compute and IO tile graph.
  snap::Graph &getVirtualGraph(VGraphId virtualGraphIndex,
                               TileSet tileSet = TileSet::Compute);

  // Return the name of the task which initializes/creates a poplar::Tensor in a
  // snap::Graph. This is NOT about creating a poplar::Program.
  PriTaskDependency taskWhichCreates(TensorId) const;

  // Return the name of the task which adds code which sets the initial
  // values of poplar::Tensor to a fragment. This IS about creating a
  // poplar::Program. For Variable Tensors, this is the Copy from Stream program
  TaskId taskWhichPopulates(TensorId) const;

  // Helper method to get the replication factor based on the user options
  unsigned getReplicationFactor() const;
  unsigned getAccumulationFactor() const;

  // If globalReplicatedGraphs are enabled then this will return an
  // offset into the global instances, otherwise 0.
  unsigned getGlobalReplicaOffset() const;

  unsigned getGlobalReplicationFactor() const;

  bool isReplicatedGraph() const;

  // The number of Poplar sequences associated with a graph.
  int getNumFragments(const Graph &graph) const;
  // Determine if any Poplar sequences associated with a graph are allocated.
  bool containsFragments(const Graph &graph) const;
  // Determine whether a specific Poplar sequence associated with a graph has
  // been allocated.
  bool containsFragment(const Graph &graph,
                        SubgraphPartIndex subgraphPart) const;

  // Ensure a specific Poplar sequence is allocated.
  void createFragment(const Graph &graph, SubgraphPartIndex subgraphPart);
  // Wrap all Poplar sequences associated with a graph in to a poplar function
  // that can be called and return them all.
  std::vector<poplar::Function> &getFragmentFunctions(const Graph &graph);
  // Wrap all Poplar sequences associated with a graph in to a poplar function
  // that can be called and return a specific one.
  poplar::Function &getFragmentFunction(const Graph &graph,
                                        SubgraphPartIndex subgraphPart);

  // A forward search of graph:
  //   - from inputs of the graph
  //   - to PopOpxs with optimised poplar calls to create the tensor,
  //     or to PopOpxs that destroy layout information of the input
  //     tensor on the output
  //   - traversing through PopOpxs that cannot create the tensor
  //     themselves, but preserve layout information from input
  //     to output tensor
  //   - tracking the route taken through the graph to the endpoints
  // Using the default arguments will return only creator candidates,
  // with each candidate's path containing only PopOpxs that need to be
  // 'unwound' to correctly lay out the input tensor
  std::vector<ICreatorCandidatePtr>
  getCreatorEndpoints(const Tensor *tensor,
                      bool excludeEndpointsFromPath = true,
                      bool includeDeadends          = false) const;

  // Get a single creator candidate for creating a tensor
  // Will throw an error if multiple candidates that do not agree are found
  std::vector<ICreatorCandidatePtr> getTensorCreators(Tensor *tensor) const;

  snap::Tensor getConst(snap::Graph &graph,
                        const poplar::Type &type,
                        const std::vector<size_t> &shape,
                        double val,
                        const poplar::DebugContext &dc = {});

  std::shared_ptr<gcl::CollectiveBalancedReorder>
  getCollectiveBalancedReorder(TensorId tensor_id);
  const gcl::CollectiveBalancedHostRearrangement &
  getCollectiveBalancedHostRearrangement(const TensorId &tensor_id) const;

  void
  setCollectiveBalancedReorder(TensorId tensor_id,
                               std::shared_ptr<gcl::CollectiveBalancedReorder>);

  const std::map<TensorId, std::shared_ptr<gcl::CollectiveBalancedReorder>> &
  getCollectiveReorders() const {
    return collectiveReorders;
  }

  snap::Tensor getScalarVariable(snap::Graph &graph,
                                 const poplar::Type &type,
                                 const poplar::DebugContext &dc = {});

  LinearMapper &getLinearMapper() { return linearMapper; }

  const liveness::LivenessAnalyzer *getLivenessAnalyzer() const {
    return livenessAnalyzer.get();
  }

  const liveness::SubgraphPartitioner *getSubgraphPartitioner() const {
    return subgraphPartitioner.get();
  }

  liveness::AliasZeroCopy *getAliasZeroCopy() const {
    return aliasZeroCopy.get();
  }

  const DeviceInfo *getDeviceInfo() { return deviceInfo.get(); }

  std::unique_ptr<PopOpx> createOpx(Op *);

  // 1-to-1 mapping between Ops and Opxs
  std::map<OpId, std::unique_ptr<PopOpx>> opxs;

  PopOpx *getOpx(OpId id) { return opxs.at(id).get(); }

  const PopOpx *getOpx(OpId id) const { return opxs.at(id).get(); }

  // Some functions useful for logging the order in which Ops are used to
  // generate poplar code / recomputed.
  //
  // The Ops in order of code generation/recompute
  const std::vector<Op *> &getMainGraphOpSeries() const;

  // index of first appearance of Op in series
  std::map<Op *, int, POpCmp> getMainGraphOpSeriesNums() const;

  // number of appearances of each Op. Expectation: Recompute Ops appear twice
  // and Checkpoint Ops appear once
  std::map<Op *, int, POpCmp> getMainGraphOpCounts() const;

  // A summary string of the Op series, with annotation for recomputation
  std::string getContextOpString(ExecutionContext context,
                                 const std::vector<TaskId> &taskOrder) const;

  bool prepareGraphHasBeenCalled() const { return prepareGraphHasBeenCalled_; }

  bool getOuterLoopFragEmpty() const { return outerLoopFragEmpty; }

  bool usingCachedExecutable() const { return usingCachedExecutable_; }

  // The ID of the poplar::Stream host->device for poplar::Tensor
  static PopStreamId h2dId(TensorId);

  // and for device->host
  static PopStreamId d2hId(TensorId, bool isAnchorStream);

  static PopStreamId gradientStoreStreamId(TensorId id);
  static PopStreamId gradientLoadStreamId(TensorId id);
  static PopStreamId weightLoadStreamId(TensorId id);

  bool hasRemoteBuffer(RemoteBufferId) const;

  const std::pair<poplar::RemoteBuffer, nonstd::optional<poplar::Tensor>> &
      getRemoteBuffer(RemoteBufferId) const;

  static const std::string getRemoteBufferName(RemoteBufferId);

  void createRemoteBuffer(RemoteBufferId, poplar::Tensor);

  poplar::RemoteBuffer &
  getOrCreateHostReduceRemoteBuffer(TensorId, TensorInfo, snap::Graph &);
  poplar::DataStream &
  insertGradientStoreStream(TensorId, TensorInfo, snap::Graph &);
  poplar::DataStream &
  insertGradientLoadStream(TensorId, TensorInfo, snap::Graph &);
  poplar::DataStream &
  insertWeightLoadStream(TensorId, TensorInfo, snap::Graph &);

  const std::vector<TensorId> &getHostReduceStreamIds() const;
  std::vector<TensorId> &getHostReduceStreamIds();

  const std::map<TensorId, poplar::RemoteBuffer> &
  getHostReduceRemoteBuffers() const;

  const std::map<TensorId, poplar::DataStream> &getFromHostStreams() const {
    return fromHostStreams;
  }

  const std::map<TensorId, poplar::DataStream> &getToHostAnchorStreams() const {
    return toHostAnchorStreams;
  }

  const std::map<TensorId, poplar::DataStream> &getToHostWeightStreams() const {
    return toHostWeightStreams;
  }
};

} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_IRLOWERING_HPP
