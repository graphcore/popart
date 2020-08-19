// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#include <popart/vendored/optional.hpp>

#include <poplar/DataStream.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>

#include <popart/aliaszerocopy.hpp>
#include <popart/devicemanager.hpp>
#include <popart/popx/creatorx.hpp>
#include <popart/popx/enigma.hpp>
#include <popart/popx/linearmapper.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/popprograms.hpp>
#include <popart/popx/poptensors.hpp>
#include <popart/popx/pritask.hpp>
#include <popart/popx/virtualgraph.hpp>

#include <set>
#include <tuple>
#include <popart/names.hpp>
// MutableVoidData is defined in here:
#include <popart/stepio.hpp>
#include <popart/stepiosplitter.hpp>
#include <popart/tensordata.hpp>

namespace popart {
namespace liveness {
class LivenessAnalyzer;
}
namespace popx {

using PopStreamId = std::string;

class Opx;
class GraphCachex;
class CollectiveBalancedReorder;

// A class containing the tensors needed to track the
// state of the pipeline
class PipelineInfo {
public:
  PipelineInfo() = default;
  PipelineInfo(int64_t _batchesPerStep,
               int64_t _gradAcclFactor,
               int64_t _maxPipelineStage,
               bool _doTraining,
               bool _doGradAccl);

  bool doTraining;
  bool doGradAccl;

  struct PipelinePhase {
    // [start, end]
    PipelineCycle start, end;
  };

  PipelinePhase fillPhase;

  // The phase between the pipeline being filled and flushed
  PipelinePhase mainPhase;

  PipelinePhase flushPhase;

  bool doStage(PipelineCycle, PipelineStage) const;
};

poplar::Type popType(const TensorInfo &);
poplar::Type popType(DataType);

enum class ToHostStreamType { NonAnchor, NonSumAnchor, SumAnchor };

class Devicex {

private:
  const Ir &_ir;

public:
  const Ir &ir() const { return _ir; }
  Devicex(const Ir &, std::shared_ptr<DeviceInfo> deviceInfo);
  ~Devicex();

private:
  void trySaveTensorTileMap() const;

  // Prepares the graph ready for poplar compilation
  void prepareGraph();

public:
  // Compile the graph and export the executable and metadata to the
  // specified paths
  void compileAndExport(const std::string &executablePath,
                        const std::string &weightsPath);

  // Compiles the graph and then prepares the streams for running on the device
  void prepare();

  void weightsFromHost();
  void remoteBufferWeightsFromHost();
  void optimizerFromHost();
  // Streams the random seed value from host, and sets the rng registers on
  // the device
  void setRandomSeedFromHost();
  const std::string cycleCountStreamId(std::string id) const;
  void instrumentWithHardwareCycleCounter(poplar::program::Sequence &,
                                          int64_t tileId = 0,
                                          std::string id = "");
  std::map<std::string, uint64_t> cycleCountTensorToHost();
  void run(IStepIO &, std::string debugName = "");

private:
  // the number of times run(IStepIO &) has been called
  int nCallsToRun{0};

  void compileAndExportExecutable(const poplar::OptionFlags &engine_options);

public:
  // device -> host stream
  void weightsToHost();
  void remoteBufferWeightsToHost();
  // device ->host stream -> specified host addresses
  void weightsToHost(const std::map<TensorId, MutableVoidData> &);

  // TODO T8229 : change these names to disambiguate
  // the source and destination

  // Write weights from (CPU end of) stream, to dst (host -> host)
  void readWeights(const IWeightsIO &dst);

  // Write weights from src to Ir Tensor memory (host -> host)
  void writeWeights(const IWeightsIO &src);

  std::string getSummaryReport(bool resetProfile = true) const;
  std::string getGraphReport(bool useCbor = false) const;
  std::string getExecutionReport(bool useCbor      = false,
                                 bool resetProfile = true) const;
  void saveTensorTileMap(const std::string &) const;
  TensorTileMap getTensorTileMap() const;
  std::string getSerializedGraph() const;

  // Return stored input tensors based on how they are allocated
  std::set<TensorId> getLinearlyCreatedInputTensors() const;
  std::set<TensorId> getEfficientlyCreatedInputTensors() const;

  PopPrograms progs;

  Opx *getOpx(OpId);
  const Opx *getOpx(OpId) const;

  poplar::Graph &graph();
  const poplar::Graph &graph() const;

  // Return virtual graph mapping to IPU virtualGraphIndex,
  // ioTileGraph selects between compute and IO tile graph.
  poplar::Graph &getVirtualGraph(VGraphId virtualGraphIndex,
                                 IsIoTile ioTileGraph = false);

  // Return the name of the task which initializes/creates a poplar::Tensor in a
  // poplar::Graph. This is NOT about creating a poplar::Program.
  std::pair<TaskId, DependencyType> taskWhichCreates(TensorId);

  // Return the name of the task which adds code which sets the initial
  // values of poplar::Tensor to a fragment. This IS about creating a
  // poplar::Program. For Variable Tensors, this is the Copy from Stream program
  TaskId taskWhichPopulates(TensorId) const;

  // PlanningCache for matmul and conv
  poplin::PlanningCache convCache;
  poplin::matmul::PlanningCache matmulCache;

  poplar::OptionFlags engineOptions, reportOptions;
  poplar::OptionFlags pooling_options;
  poplar::OptionFlags lstmOptions;
  poplar::OptionFlags gclOptions;

  PopTensors tensors;

  // Helper method to get the replication factor based on the user options
  unsigned getReplicationFactor() const;
  unsigned getAccumulationFactor() const;

  // If globalReplicatedGraphs are enabled then this will return an
  // offset into the global instances, otherwise 0.
  unsigned getReplicaOffset() const;

  unsigned getGlobalReplicationFactor() const;

  bool isReplicatedGraph() const;

  PipelineInfo pipelineInfo() const;

  bool containsFragment(const Graph &scope) const;
  void createFragment(const Graph &);

  poplar::Function &getFragmentFunction(const Graph &called_graph);

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

  // Get a single creator candidate for creating a tensor
  // Will throw an error if multiple candidates that do not agree are found
  ICreatorCandidatePtr getTensorCreator(Tensor *tensor) const;

  bool isEngineLoaded() const;
  void setEngineIsLoaded(bool isLoaded);

  // Compile-time poplar tensors used to determine sampling of random
  // numbers across tiles. Combined with the random seed and seedModifier,
  // this ensures that the same random mask is generated for fwd and bwd
  // dropout ops in the same layer
  // TODO: remove from this class, see T15790
  std::map<uint32_t, poplar::Tensor> dropoutReferenceTensors;

  poplar::Tensor getConst(poplar::Graph &graph,
                          const poplar::Type &type,
                          const std::vector<size_t> &shape,
                          double val,
                          const std::string &name);

  bool hasRemoteBuffer(RemoteBufferId) const;

  const std::pair<poplar::RemoteBuffer, nonstd::optional<poplar::Tensor>> &
      getRemoteBuffer(RemoteBufferId) const;

  const std::string getRemoteBufferName(RemoteBufferId) const;

  void createRemoteBuffer(RemoteBufferId, poplar::Tensor);

  std::shared_ptr<CollectiveBalancedReorder>
  getCollectiveBalancedReorder(TensorId tensor_id);
  void setCollectiveBalancedReorder(TensorId tensor_id,
                                    std::shared_ptr<CollectiveBalancedReorder>);

  poplar::Tensor getScalarVariable(poplar::Graph &graph,
                                   const poplar::Type &type,
                                   const std::string &name);

  PopStreamId gradientStoreStreamId(TensorId id) const;
  PopStreamId gradientLoadStreamId(TensorId id) const;
  PopStreamId weightLoadStreamId(TensorId id) const;

  poplar::RemoteBuffer &
  getOrCreateHostReduceRemoteBuffer(TensorId, TensorInfo, poplar::Graph &);
  poplar::DataStream &
  insertGradientStoreStream(TensorId, TensorInfo, poplar::Graph &);
  poplar::DataStream &
  insertGradientLoadStream(TensorId, TensorInfo, poplar::Graph &);
  poplar::DataStream &
  insertWeightLoadStream(TensorId, TensorInfo, poplar::Graph &);

  const std::vector<TensorId> &getHostReduceStreamIds() const;
  std::vector<TensorId> &getHostReduceStreamIds();

  const std::map<TensorId, poplar::RemoteBuffer> &
  getHostReduceRemoteBuffers() const;

  void connectStreamToCallback(const std::string &streamHandle,
                               std::function<void(void *)> callback,
                               unsigned index);

  LinearMapper &getLinearMapper() { return linearMapper; }

  void copyFromRemoteBuffer(const PopStreamId buffer,
                            void *w,
                            int repeat_index,
                            unsigned replication_index = 0);

  void copyToRemoteBuffer(void *w,
                          const PopStreamId buffer,
                          int repeat_index,
                          unsigned replication_index = 0);

  const liveness::LivenessAnalyzer *getLivenessAnalyzer() const {
    return livenessAnalyzer.get();
  }

  liveness::AliasZeroCopy *getAliasZeroCopy() const {
    return aliasZeroCopy.get();
  }

  const DeviceInfo *getDeviceInfo() { return deviceInfo.get(); }

private:
  std::unique_ptr<poplar::Graph> pGraph{nullptr};
  std::unique_ptr<poplar::Engine> pEngine{nullptr};

  std::vector<VirtualGraph> virtualGraphs;

  std::shared_ptr<DeviceInfo> deviceInfo;

  // Non-const tensors used to keep track of batch count, modulo the return
  // period
  std::map<ReturnPeriod, poplar::Tensor> batchCountingTensors;
  std::map<ReturnPeriod, poplar::Tensor> batchCountCheckingTensors;

  // Map tensors evenly across all tiles
  LinearMapper linearMapper;

  poplar::Tensor randomSeedTensor;

  PipelineInfo pInfo;

  std::map<PipelineStage, VGraphId> getPipelineToVGraphIdMap() const;

  // Task to create a poplar::Tensor from nothing, choosing
  // the correct create call (createWeights, addLinearly, etc)
  PriTask initTensorTask(Tensor *);
  PriTask initTensorByCloningTask(Op *op, TensorId srcId, TensorId dstId);
  PriTask initTensorByAliasingTask(Op *op, TensorId srcId, TensorId dstId);
  TaskId initTensorTaskId(TensorId) const;
  bool tryInitTensorByPostIRAliasing(TensorId dstId);

  PriTask initRandomSeed();
  TaskId initRandomSeedTaskId() const;
  void connectRandomSeedStream();

  PriTask setInitTensorValTask(Tensor *);
  TaskId setInitTensorValTaskId(TensorId) const;

  // Task to create a poplar::Stream to write to poplar::Tensor
  // C++ Note: if a lambda function which modifies `this' is created
  // it must be const w.r.t this, even if it not run
  PriTask streamFromHostTask(Tensor *);
  TaskId streamFromHostTaskId(TensorId) const;

  // Task to append a Copy from poplar::Stream to poplar::Tensor
  PriTask fromHostTask(Tensor *tensor,
                       poplar::program::Sequence &streamSq) const;

  TaskId fromHostTaskId(TensorId) const;

  // Task to create a poplar::Stream to write from poplar::Tensor to host
  PriTask streamToHostTask(Tensor *, bool isAnchorStream);
  TaskId streamToHostTaskId(TensorId, bool isAnchorStream) const;

  poplar::program::Sequence &getAnchorReturnFragment(Tensor *tensor);

  // Task to append a Copy to poplar::Stream from poplar::Tensor
  PriTask toHostTask(Tensor *tensor,
                     poplar::program::Sequence &,
                     ToHostStreamType) const;
  TaskId toHostTaskId(TensorId, bool isAnchorStream) const;

  // Task to create an accumulator and scaleAddto to a poplar::Tensor to be
  // Copied on the final batch per step
  PriTask anchorReturnTypeSumTask(Tensor *tensor,
                                  poplar::program::Sequence &sq);
  TaskId anchorSumTaskId(const TensorId &) const;

  // Task to create poplar::Tensors from nothing, specifically for
  // use in keeping track of the batch count
  PriTask initBatchCounterTensorsTask();
  TaskId initBatchCounterTensorsTaskId() const;

  // Task to add a program to increment and check the batch count
  PriTask updateBatchCountTask(poplar::program::Sequence &sq);
  TaskId updateBatchCountTaskId() const;

  // Task to append a Copy to poplar::Stream from poplar::Tensor every
  // N batches
  PriTask toHostEveryNBatchesTask(Tensor *tensor,
                                  ReturnPeriod N,
                                  poplar::program::Sequence &);

  PriTask initAndUpdatePipelineStashIndicesTask();

  PriTask opTask(Op *, double priority, TaskId prevOpTaskId);
  void opTaskFunc(TaskId taskId, Op *, SequenceMap &seqs);
  void pipelinedOpTaskFunc(TaskId taskId, Op *, SequenceMap &seqs);
  void growOpx(Opx *, poplar::program::Sequence &);

  TaskId opTaskId(Op *) const;

  void addOpTasks(PriTasks &);

  TaskId pipelinedCopyTaskId(Op *op) const;

  void addPipelinedCopyTasks(PriTasks &);

  PriTask pipelinedCopyTask(Op *, TaskId prevTaskId);

  bool doRearrangeOnHost(Tensor *tensor) const;

  void initPoplarGraph();

public:
  // The ID of the poplar::Stream host->device for poplar::Tensor
  PopStreamId h2dId(TensorId) const;

  // and for device->host
  PopStreamId d2hId(TensorId, bool isAnchorStream) const;

  std::unique_ptr<Opx> createOpx(Op *);

  // 1-to-1 mapping between Ops and Opxs
  std::map<OpId, std::unique_ptr<Opx>> opxs;

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
  std::string getMainGraphOpString(const std::vector<TaskId> &taskOrder) const;

  bool prepareHasBeenCalled() const { return prepareHasBeenCalled_; }

private:
  std::map<TaskId, std::vector<Op *>> mainGraphOpRegistry;
  std::map<TaskId, std::vector<Op *>> requiredRecomputes;

  void verifyTaskOrder(const std::vector<TaskId> &taskOrder) const;

  // We have datastreams which are created during the prepare phase and
  // associated with the stream call back
  // Then when run is called the data streams are associated with the
  // step oi class

  class Datastream {

  protected:
    Tensor *tensor;
    PopStreamId streamId;

    // This is per data stream to allow for different stepio
    // configurations per data stream.
    // Q : Is there a better type than a pointer?
    IStepIO *io;

  public:
    Datastream(Tensor *ten, PopStreamId s);

    void setStepIO(IStepIO *v) { io = v; }

    TensorId getTensorId();
  };

  // host to device data stream
  class InputDatastream : public Datastream {
  public:
    InputDatastream(Tensor *t, PopStreamId s);

    // Called to read data from an input stream
    void read(void *ptr);

    // Called to prefetch data from an input stream
    // return true is there is data prefetch else false
    bool readPrefetch(void *ptr);

    // Called to indicate the data has been comsumed
    // by poplar
    void readComplete();
  };

  class PrefetchCallback : public poplar::StreamCallback {
  public:
    PrefetchCallback(std::shared_ptr<InputDatastream> ds_);

    poplar::StreamCallback::Result prefetch(void *dest) noexcept override;
    void fetch(void *dest) noexcept override;
    void complete() noexcept override;

  private:
    std::shared_ptr<InputDatastream> ds;
  };

  // device to host data stream
  class OutputDatastream : public Datastream {
  public:
    OutputDatastream(Tensor *t, PopStreamId s);
    void write(void *ptr);
  };

  // Splits one IStepIO into one for each replica.
  std::unique_ptr<StepIOSplitter> stepIoSplitter;

  // Map from TensorId,replicationIndex to the data streams
  using StreamId = std::tuple<TensorId, unsigned>;
  std::map<StreamId, std::shared_ptr<InputDatastream>> inputStreams;
  std::map<StreamId, std::shared_ptr<OutputDatastream>> outputStreams;

  // T11630: should we combine the above with the below

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

  // Collective balanced reordering information for replicated ops
  std::map<TensorId, std::shared_ptr<CollectiveBalancedReorder>>
      collectiveReorders;

  // Streams for doing allreduce on host side

  std::map<TensorId, poplar::RemoteBuffer> hostReduceRemoteBuffers;
  std::map<TensorId, poplar::DataStream> toHostGradientStreams;
  std::map<TensorId, poplar::DataStream> fromHostGradientStreams;
  std::map<TensorId, poplar::DataStream> fromHostWeightLoadStreams;

  std::vector<TensorId> hostReduceStreamIds;

  // Q: Consider replacing the d2h weight buffer with a data stream as
  // done for inputs
  std::map<TensorId, std::vector<char>> d2hWeightBuffers;
  std::map<TensorId, std::vector<char>> chBuffers;

  // Buffers for storing the hardware cycle count
  std::map<std::string, uint64_t> cycleCount;

  // Wrapper for calls to poplar Engine API calls: loading
  // engine onto the poplar device and connecting streams.
  // Must be called before running a poplar program with a
  // call to this Devicex's engine.
  void loadEngineAndConnectStreams();

  // We may have prefetched data ready to be fed into the model, but we have
  // provided a new buffer which we want to be fetched. We invalidate the
  // prefetch by reconnecting the datastreams before each program run.
  void reconnectInputStreams();

  // Is this Devicex's engine the last to have been loaded onto
  // deviceInfo's device?
  // Becomes true once loadEngineAndConnectStreams() is called.
  // Becomes 'false' if another engine has been loaded after
  // loadEngineAndConnectStreams() has been called. This is
  // different to 'prepareHasBeenCalled_', which, once true,
  // is always true
  bool engineIsLoaded = false;

  // Wrapper function that checks the calling devicex was the
  // last to have loaded its engine to deviceInfo's device
  void run(PopPrograms::ProgramIndex ind, std::string debugName);

  void hostStreamToHost(const MutableVoidData &mv_data, TensorId id);

  // Call hostToHostStream on all the Tensors in pir->dataStreamTensors()
  void anchorsHostToHostStreams(IStepIO &stepio);

  // Call hostStreamToHost in all the Tensors in pir->dataFlow.anchors()
  void anchorsHostFromHostStreams(IStepIO &stepio);

  template <typename T> void setInitVal(Tensor *tensor);
  void setInitValHalf(Tensor *tensor);

  // Either return the executable in cachedExecutable
  // or compile `rootGraph' and try to save the generated executable before
  // returning it. After calling `getExecutable', `cachedExecutable' will always
  // be set to `nonstd::nullopt'.
  poplar::Executable getExecutable();

  // Try to save the argument executable to a file at
  // `ir().getSessionOptions().cachePath'.
  void trySaveExecutable(poplar::Executable &);

  // Try to load a poplar::Executable from a file at
  // `ir().getSessionOptions().cachePath'. If successful,
  // `this->cachedExecutable' will be set else, `this->cachedExecutable' will
  // remain set to `nonstd::nullopt'.
  void tryLoadExecutable();

  std::string getPoplarCachePath();
  std::string getPopartCachePath();

  void setFloatingPointBehaviour(poplar::Graph &graph);
  void setStochasticRoundingBehaviour(poplar::Graph &graph);

  void doProfileChecks() const;

  // Store input tensors based on how they are allocated
  std::set<TensorId> linearlyCreatedInputTensors;
  std::set<TensorId> efficientlyCreatedInputTensors;

  bool prepareHasBeenCalled_;
  bool prepareGraphHasBeenCalled_;

  nonstd::optional<poplar::Executable> cachedExecutable;
  bool usingCachedExecutable = false;

  // Option to trace the opx execution using printTensor. This can be useful in
  // determining where an exception has occurred. It is enabled by setting
  // POPART_OPX_TRACE environment variable to "1"
  bool opxTrace = false;
  poplar::Tensor opxTraceTensor;

  // This keeps track of whether there the accumulateOuterFragment is empty
  // TODO T12001 a class which encapsulates framgments which has this attribute.
  bool outerLoopFragEmpty = true;

  // Helper class to analyze the global IR schedule and tensor liveness
  std::unique_ptr<liveness::LivenessAnalyzer> livenessAnalyzer;

  // Helper class to reuse tensors and call subgraphs by reference
  std::unique_ptr<liveness::AliasZeroCopy> aliasZeroCopy;

public:
  bool getOuterLoopFragEmpty() const { return outerLoopFragEmpty; }
};

} // namespace popx
} // namespace popart

#endif
