// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_IRLOWERING_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_IRLOWERING_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <poplar/DataStream.hpp>
#include <poplar/Executable.hpp>
#include <poplar/FunctionBufferMappingType.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/ReplicatedStreamMode.hpp>
#include <poplar/Type.hpp>
#include <poplar/exceptions.hpp>
#include <popart/names.hpp>
#include <popart/popx/exchangebundle.hpp>
#include <popart/popx/inittensor.hpp>
#include <popart/popx/linearmapper.hpp>
#include <popart/popx/namesx.hpp>
#include <popart/popx/popprograms.hpp>
#include <popart/popx/poptensors.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/popx/replicatedtensorshardingbundle.hpp>
#include <popart/popx/virtualgraph.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/preparedtensor.hpp"
#include "popart/popx/viewchangers.hpp"
#include "popart/taskid.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorlocation.hpp"

namespace poplar {
class Target;
} // namespace poplar
namespace poplin {
struct ConvParams;
struct MatMulParams;
} // namespace poplin
namespace poplar {
class Function;
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class DeviceInfo;
class Graph;
class Ir;
class Tensor;
struct POpCmp;
struct SessionOptions;
class TensorInfo;
class ProfileCacher;

namespace liveness {
// Forward declaration.
class LivenessAnalyzer;
class AliasZeroCopy;
class SubgraphCopyingStrategy;
class SubgraphPartitioner;
} // namespace liveness
namespace popx {
class PopOpx;
namespace serialization {
// Forward declaration.
class Reader;
} // namespace serialization

// Forward declaration.
class RngStateLowering;
class OpxState;

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

class Devicex;

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
  std::unique_ptr<poplar::Graph> pGraph;

  std::vector<VirtualGraph> virtualGraphs;

  std::shared_ptr<DeviceInfo> deviceInfo;

  ProgressLogger progressLogger;

  std::map<PipelineStage, VGraphId> getPipelineToVGraphIdMap() const;

  std::map<std::pair<ExecutionContext, TaskId>, std::vector<Op *>>
      contextOpRegistry;
  std::map<TaskId, std::vector<Op *>> requiredRecomputes;

  /**
   * Container for replicated tensor sharding related lowering state
   */
  ReplicatedTensorShardingBundle replicatedTensorShardingBundle;

  /**
   * Container for stream and remote buffer related lowering state
   */
  ExchangeBundle exchangeBundle;

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

  // Keep track of which tensors already have been printed
  std::unordered_set<TensorId> printedTensorIds;

  // Keep track of Pipeline IPU copy source and targets
  std::map<OpId, PreparedCopyTensors> pipelineIpuCopySrcDst;

  // This keeps track of whether there the accumulateOuterFragment is empty
  // TODO T12001 a class which encapsulates fragments which has this attribute.
  bool outerLoopFragEmpty = true;

  // Helper class to determine where to move input/output copies.
  std::unique_ptr<liveness::SubgraphCopyingStrategy> subgraphCopyingStrat;

  // Helper class to analyze the global IR schedule and tensor liveness
  std::unique_ptr<liveness::LivenessAnalyzer> livenessAnalyzer;

  // Helper class to reuse tensors and call subgraphs by reference
  std::unique_ptr<liveness::AliasZeroCopy> aliasZeroCopy;

  // Helper class to interpret results of liveness analyzer.
  std::unique_ptr<liveness::SubgraphPartitioner> subgraphPartitioner;

  // Non-const tensors used to keep track of batch count, modulo the return
  // period
  std::map<ReturnPeriod, poplar::Tensor> batchCountingTensors;
  std::map<ReturnPeriod, poplar::Tensor> batchCountCheckingTensors;

  // Implicit pipelining index tensors (i.e. stash and restore counters)
  std::vector<poplar::Tensor> pipelineIndexTensors;

  // Map tensors evenly across all tiles
  LinearMapper linearMapper;

  poplar::Tensor randomSeedTensor;

  // TODO T11630: Combine the inputStreams/outputStreams with the
  // fromHostStreams/toHostAnchorStreams streams?

  //  poplar::Streams for poplar::Tensors,
  //  1) from host to device;
  std::map<TensorId, poplar::DataStream> fromHostStreams;

  // and 2) from device to host;
  std::map<TensorId, poplar::DataStream> toHostAnchorStreams;
  std::map<TensorId, poplar::DataStream> toHostWeightStreams;

  // Streams for doing allreduce on host side
  std::map<TensorId, poplar::DataStream> toHostGradientStreams;
  std::map<TensorId, poplar::DataStream> fromHostGradientStreams;
  std::map<TensorId, poplar::DataStream> fromHostWeightLoadStreams;

  // The maximum number of inputs on any Op
  int maxOpInputs;

  // Container for temporary Opx states during lowering
  std::map<OpId, std::unique_ptr<OpxState>> opxState;

  unsigned tileCounterGraphConstVar;
  int tileCounterGraphScalarVar;

  // Map of custom programs that can be run
  std::map<std::string, unsigned> programHandleIndexMap;

  // Unique ptr so that rngstatelowering.hpp can be a private header.
  std::unique_ptr<RngStateLowering> rngStateLowering;

  // Used for opx creation
  Devicex *dv_p = nullptr;

  void setMetadataFromIr();

  void verifyTaskOrder(const std::vector<TaskId> &taskOrder) const;

  // Task to create a poplar::Tensor from nothing, choosing
  // the correct create call (createWeights, addLinearly, etc)
  // If dependencyFree is true, the creator must not depend on any other tensor
  // having been created. This can be (sparingly) used to resolve cyclic
  // dependencies
  InitTensorPtrs
  getInitTensorCreators(Tensor *,
                        RequireParallelWritable requireParallelWritable,
                        bool dependencyFree = false);

  // Task to create a poplar::Tensor with methods defined by InitTensorPtrs
  PriTask initTensorTask(InitTensorPtrs inits);

  static TaskId initTensorTaskId(TensorId);

  PriTask initRandomSeed(Tensor *tensor);
  static TaskId initRandomSeedTaskId();

  PriTask setInitTensorValTask(Tensor *);
  static TaskId setInitTensorValTaskId(TensorId);

  // Task to create a poplar::Stream to write to poplar::Tensor
  // C++ Note: if a lambda function which modifies `this' is created
  // it must be const w.r.t this, even if it not run
  PriTask streamFromHostTask(TensorId streamTensorId,
                             std::vector<Tensor *> tensors);
  static TaskId streamFromHostTaskId(TensorId);

  // Task to append a Copy from poplar::Stream to poplar::Tensor
  PriTask fromHostTask(Tensor *tensor,
                       poplar::program::Sequence &streamSq,
                       bool isUpdate = false);

  static TaskId fromHostTaskId(TensorId, bool isUpdate = false);

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

  // The tasks associated with lowering an operation
  std::vector<PriTask> opTasks(Op *, double priority, TaskId prevOpTaskId);

  void opTaskFunc(TaskId taskId, Op *, SequenceMap &seqs);
  void pipelinedOpTaskFunc(TaskId taskId, Op *, SequenceMap &seqs);

  // PopOpx version of growOpx(Opx *, SequenceMap::SequenceInterval seqInterval)
  // for code transition, see above.
  [[deprecated("Temporary function to help with code transition. Internal use "
               "only.")]] void
  growPopOpxCall(PopOpx *popOpx, SequenceMap::SequenceInterval seqInterval);

  // The name of the task associated with growing an operation
  TaskId opTaskId(Op *) const;

  // The name of the task associated with an operation creating an output tensor
  TaskId opTensorTaskId(Op *, Tensor *) const;

  // The name of the task associated with a part of an operation
  TaskId opPartTaskId(Op *, OpxGrowPartId) const;

  void addOpTasks(PriTasks &);

  static TaskId pipelinedCopyTaskId(Op *op);

  void addPipelinedCopyTasks(PriTasks &);

  PriTask pipelinedCopyTask(Op *, TaskId prevTaskId);

  // Determine stream properties.
  poplar::ReplicatedStreamMode getReplicatedStreamMode(Tensor *tensor) const;
  unsigned getBufferingDepth(Tensor *tensor) const;

  void initPoplarGraph();

  template <typename T>
  void setInitVal(Tensor *tensor, DataType dstType = DataType::UNDEFINED);
  void setInitValHalf(Tensor *tensor);

  void setFloatingPointBehaviour(poplar::Graph &graph);
  void setStochasticRoundingBehaviour(poplar::Graph &graph);

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
  PopPrograms progs_;

public:
  IrLowering(const Ir &,
             std::shared_ptr<DeviceInfo> deviceInfo,
             bool prepareGraphHasBeenCalled = false);
  virtual ~IrLowering();

  const Ir &ir() const { return _ir; }

  void growOpx(Opx *, SequenceMap::SequenceInterval seqInterval);

  void growOpxCall(Opx *, SequenceMap::SequenceInterval seqInterval);

  poplar::OptionFlags pooling_options;
  poplar::OptionFlags lstmOptions;
  poplar::OptionFlags matmulOptions;
  poplar::OptionFlags gclOptions;
  poplar::OptionFlags engineOptions;
  poplar::OptionFlags reportOptions;

  void setDevicex(Devicex *d) { dv_p = d; }
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

  bool
  tryInitTensorByPostIRAliasing(TensorId dstId,
                                RequireParallelWritable requireParallelWritable,
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

  const PopPrograms &progs() const { return progs_; }
  PopPrograms &progs() { return progs_; }

  void instrumentWithHardwareCycleCounter(poplar::program::Sequence &,
                                          int64_t tileId = 0,
                                          std::string id = "");

  poplar::Graph &graph() {
    if (pGraph == nullptr) {
      throw error(
          "poplar::Graph is null when the lowering state is deserialized");
    }
    return *pGraph;
  }
  const poplar::Graph &graph() const {
    if (pGraph == nullptr) {
      throw error(
          "poplar::Graph is null when the lowering state is deserialized");
    }
    return *pGraph;
  }
  // Prepares the graph ready for poplar compilation
  void prepareGraph();

  // Load a poplar::Executable using reader object
  // and set `this->cachedExecutable'.
  void loadPoplarExecutable(serialization::Reader &reader);

  // Either return the executable in cachedExecutable
  // or compile `rootGraph' and try to save the generated executable before
  // returning it. After calling `getExecutable', `cachedExecutable' will always
  // be set to `nonstd::nullopt'.
  poplar::Executable getExecutable(const ProfileCacher &ProfileCacher);

  // Return the name that will be passed when compiling the executable
  std::string getPoplarGraphDebugName();

  std::string getSerializedGraph() const;

  // Return virtual graph mapping to IPU virtualGraphIndex,
  // ioTileGraph selects between compute and IO tile graph.
  poplar::Graph &getVirtualGraph(VGraphId virtualGraphIndex,
                                 TileSet tileSet = TileSet::Compute);

  // Return the name of the task which initializes/creates a poplar::Tensor in a
  // poplar::Graph. This is NOT about creating a poplar::Program.
  PriTaskDependency taskWhichCreates(TensorId) const;

  // Return the name of the task which adds code which sets the initial
  // values of poplar::Tensor to a fragment. This IS about creating a
  // poplar::Program. For Variable Tensors, this is the Copy from Stream program
  TaskId taskWhichPopulates(TensorId) const;

  // Obtain a dependency-free creator task for a tensor.
  // This is useful for resolving cyclic dependencies with tensor creation.
  PriTask getDependencyFreeInitTensorCreatorTask(const TensorId &);

  // Helper method to get the replication factor based on the user options
  unsigned getReplicationFactor() const;
  unsigned getAccumulationFactor() const;

  // If globalReplicatedGraphs are enabled then this will return an
  // offset into the global instances, otherwise 0.
  unsigned getGlobalReplicaOffset() const;

  unsigned getGlobalReplicationFactor() const;

  bool isReplicatedGraph() const;

  // Determine if a tensor-related stream requires rearranging on the host
  // - Overlapping IO and Compute is more likely to occur if tensors are
  //   rearranged on the host.
  // - Rearrange on the host uses less IPU memory but may require more cycles
  // - Rearrange on the device uses more IPU memory but may require fewer cycles
  bool doRearrangeOnHost(Tensor *tensor) const;

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

  /**
   * Add a vector of pairs {f, buffer} for a given graph id,
   * FunctionBufferMappingType pair. This is enough for an
   * [Internal|External]CodeCopy op to move code from the buffer in to the
   * function. Note the subgraphpartitioner may have split this into multiple
   * functions, so we require a vector of these for each graph.
   *
   * \param gid The graph id to add the functions and buffers for.
   * \param fbmt The FunctionBufferMappingType to add the vector for.
   */
  void addFunctionBuffers(const GraphId gid,
                          poplar::FunctionBufferMappingType fbmt);

  // Shorthand storage type for storing functionbuffers.
  using FunctionBuffers =
      std::vector<std::pair<const poplar::Function, poplar::FunctionBuffer>>;
  /**
   * Get the Function Buffers for the given GraphId and
   * FunctionBufferMappingType. Wrapper around popprograms function.
   *
   * \param gid The GraphId to lookup.
   * \param fbmt The FunctionBufferMappingType to lookup.
   * \returns FunctionBuffers the vector of functions and buffers.
   */
  FunctionBuffers getFunctionBuffer(const GraphId gid,
                                    poplar::FunctionBufferMappingType fbmt) {
    return progs().getFunctionBuffer(gid, fbmt);
  }

  /**
   * Returns true if a functionBuffer vector exists for the given graphId /
   * FunctionBufferMappingType. Wrapper around popprograms function.
   *
   * \param gid The graph id to lookup.
   * \param fbmt The FunctionBufferMappingType to lookup.
   * \returns true If pairs exist.
   * \returns false Otherwise.
   */
  bool hasFunctionBuffer(const GraphId gid,
                         poplar::FunctionBufferMappingType fbmt) {
    return progs().hasFunctionBuffer(gid, fbmt);
  }

  // A forward search of graph:
  //   - from inputs of the graph
  //   - to Opxs with optimised poplar calls to create the tensor,
  //     or to Opxs that destroy layout information of the input
  //     tensor on the output
  //   - traversing through Opxs that cannot create the tensor
  //     themselves, but preserve layout information from input
  //     to output tensor
  //   - tracking the route taken through the graph to the endpoints
  // Using the default arguments will return only creator candidates,
  // with each candidate's path containing only Opxs that need to be
  // 'unwound' to correctly lay out the input tensor
  std::vector<ICreatorCandidatePtr>
  getCreatorEndpoints(const Tensor *tensor,
                      bool excludeEndpointsFromPath = true,
                      bool includeDeadends          = false) const;

  // Remove any creators which are have dependencies on other tensors
  static void removeNonDependencyFreeCreators(
      std::vector<ICreatorCandidatePtr> &candidates);

  // Get a single creator candidate for creating a tensor
  // Will throw an error if multiple candidates that do not agree are found
  std::vector<ICreatorCandidatePtr>
  getTensorCreators(const Tensor *tensor, bool dependencyFree) const;

  poplar::Tensor getConst(poplar::Graph &graph,
                          const poplar::Type &type,
                          const std::vector<size_t> &shape,
                          double val,
                          const poplar::DebugContext &dc = {});

  const ReplicatedTensorShardingBundle &
  getReplicatedTensorShardingBundle() const {
    return replicatedTensorShardingBundle;
  }

  ReplicatedTensorShardingBundle &getReplicatedTensorShardingBundle() {
    return replicatedTensorShardingBundle;
  }

  poplar::Tensor getScalarVariable(poplar::Graph &graph,
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

  const DeviceInfo *getDeviceInfo() const { return deviceInfo.get(); }
  void setDeviceInfo(std::shared_ptr<DeviceInfo> deviceInfo_) {
    deviceInfo = std::move(deviceInfo_);
  }

  std::unique_ptr<Opx> createOpx(Op *);

  // 1-to-1 mapping between Ops and Opxs
  std::map<OpId, std::unique_ptr<Opx>> opxs;

  Opx *getOpx(OpId id) { return opxs.at(id).get(); }

  const Opx *getOpx(OpId id) const { return opxs.at(id).get(); }

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

  poplar::DataStream &
  insertGradientStoreStream(TensorId, TensorInfo, poplar::Graph &);
  poplar::DataStream &
  insertGradientLoadStream(TensorId, TensorInfo, poplar::Graph &);
  poplar::DataStream &
  insertWeightLoadStream(TensorId, TensorInfo, poplar::Graph &);

  void addPipelineIndexTensor(const poplar::Tensor &tensor) {
    pipelineIndexTensors.push_back(tensor);
  }

  /**
   * Get the exchange bundle containing stream and remote buffer data structures
   * \return Exchange bundle
   */
  ExchangeBundle &getExchangeBundle() { return exchangeBundle; }

  /**
   * Get the exchange bundle containing stream and remote buffer data structures
   * \return Exchange bundle
   */
  const ExchangeBundle &getExchangeBundle() const { return exchangeBundle; }

  const std::vector<poplar::Tensor> getPipelineIndexTensors() {
    return pipelineIndexTensors;
  }

  const std::map<TensorId, poplar::DataStream> &getFromHostStreams() const {
    return fromHostStreams;
  }

  const std::map<TensorId, poplar::DataStream> &getToHostAnchorStreams() const {
    return toHostAnchorStreams;
  }

  const std::map<TensorId, poplar::DataStream> &getToHostWeightStreams() const {
    return toHostWeightStreams;
  }

  // Construct and get a temporary, modifiable Opx state container,
  // for growing Opxs in parts
  template <class T> T *getOpxState(OpId opid) {
    auto it = opxState.find(opid);
    if (it == opxState.end()) {
      // Public header, pre C++14 construction of unique ptr
      opxState[opid] = std::unique_ptr<T>(new T());
    }
    return static_cast<T *>(opxState.at(opid).get());
  }

  void setProgramHandleIndexMap(
      const std::map<std::string, unsigned> &programHandleIndexMap_) {
    programHandleIndexMap = programHandleIndexMap_;
  }

  const std::map<std::string, unsigned> &getProgramHandleIndexMap() const {
    return programHandleIndexMap;
  }
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_IRLOWERING_HPP_
