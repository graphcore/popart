#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#include <boost/optional.hpp>

#include <poplar/DataStream.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>

#include <popart/devicemanager.hpp>
#include <popart/popx/enigma.hpp>
#include <popart/popx/linearmapper.hpp>
#include <popart/popx/poplaroptionsx.hpp>
#include <popart/popx/popprograms.hpp>
#include <popart/pritask.hpp>

#include <set>
#include <popart/names.hpp>
// MutableVoidData is defined in here:
#include <popart/tensordata.hpp>

using boost::optional;

namespace popart {
namespace popx {

using PopStreamId = std::string;

class Opx;
class GraphCachex;

// A class containing the tensors needed to track the
// state of the pipeline
class PipelineInfo {
public:
  PipelineInfo() = default;
  PipelineInfo(int _batchesPerStep,
               int _gradAcclFactor,
               int _numPipelineStages,
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

struct ICreatorCandidate;
using ICreatorCandidatePtr = std::shared_ptr<ICreatorCandidate>;

// An interface for a potential creator of a tensor
struct ICreatorCandidate {

  // A bundle struct to represent the path a tensor
  // takes through an Opx
  struct OpxInAndOutIndex {
    OpxInAndOutIndex(const Opx *opx_, InIndex inIndex_, OutIndex outIndex_)
        : opx(opx_), inIndex(inIndex_), outIndex(outIndex_) {}
    OpxInAndOutIndex() = default;

    const Opx *opx;
    InIndex inIndex;
    OutIndex outIndex;
  };

  ICreatorCandidate(int, const Opx *, std::vector<OpxInAndOutIndex> path);
  virtual ~ICreatorCandidate() = default;

  // Create's a input tensor
  virtual poplar::Tensor createInput(const std::string &name) = 0;

  // Returns the list of tensors that must be created before this one
  virtual std::vector<TensorId> mustExistBeforeCreate() = 0;

  virtual bool createsEquivalent(const ICreatorCandidatePtr other) = 0;

  virtual double getMaxCreatorPriority() = 0;

  virtual std::string str() = 0;

  // Returns the unwind path from the tensor to the creator
  std::vector<OpxInAndOutIndex> getPathFromInput() { return pathFromInput; }
  void setPathFromInput(std::vector<OpxInAndOutIndex> &value) {
    pathFromInput = value;
  }

  int getIndex() const { return index; }
  const Opx *getOpx() const { return opx; }

protected:
  poplar::Tensor unwind(poplar::Tensor i);
  std::vector<OpxInAndOutIndex> pathFromInput;

private:
  int index;
  const Opx *opx;
};

class InputCreatorCandidate : public ICreatorCandidate {
public:
  InputCreatorCandidate(int, const Opx *, std::vector<OpxInAndOutIndex>);
  InputCreatorCandidate()                   = default;
  virtual ~InputCreatorCandidate() override = default;

  poplar::Tensor createInput(const std::string &name) override;

  std::vector<TensorId> mustExistBeforeCreate() override;

  double getMaxCreatorPriority() override;

  bool createsEquivalent(const ICreatorCandidatePtr other) override;

  virtual std::string str() override;
};

class InputMultiCreatorCandidate : public ICreatorCandidate {
public:
  InputMultiCreatorCandidate(int,
                             const Opx *,
                             std::vector<OpxInAndOutIndex> path);
  virtual ~InputMultiCreatorCandidate() override = default;

  poplar::Tensor createInput(const std::string &name) override;
  std::vector<TensorId> mustExistBeforeCreate() override;

  double getMaxCreatorPriority() override;

  bool createsEquivalent(const ICreatorCandidatePtr other) override;

  virtual std::string str() override;

  void addCreateorCandidate(ICreatorCandidatePtr c) { candidates.push_back(c); }

private:
  std::vector<ICreatorCandidatePtr> candidates;
};

class PopTensors {
public:
  PopTensors(const Ir &);
  void insert(TensorId, const poplar::Tensor &);
  const poplar::Tensor &get(TensorId) const;
  bool contains(TensorId) const;
  const std::map<TensorId, poplar::Tensor> &getTensors() const;

private:
  std::map<TensorId, poplar::Tensor> tensors_;
  const Ir &ir;
};

class Devicex {

private:
  const Ir &_ir;

public:
  const Ir &ir() const { return _ir; }
  Devicex(const Ir &, std::shared_ptr<DeviceInfo> deviceInfo);
  ~Devicex();
  void prepare();
  void weightsFromHost();
  void optimizerFromHost();
  // Streams the random seed value from host, and sets the rng registers on
  // the device
  void setRandomSeedFromHost();

  void run(IStepIO &);

  // device -> host stream
  void weightsToHost();
  // device ->host stream -> specified host addresses
  void weightsToHost(const std::map<TensorId, MutableVoidData> &);

  // TODO T8229 : change these names to disambiguate
  // the source and destination

  // Write weights from (CPU end of) stream, to dst (host -> host)
  void readWeights(const IWeightsIO &dst);

  // Write weights from src to Ir Tensor memory (host -> host)
  void writeWeights(const IWeightsIO &src);

  std::string getSummaryReport() const;
  std::string getGraphReport(bool use_cbor = false) const;
  std::string getExecutionReport(bool use_cbor = false) const;
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

  poplar::Graph &getVirtualGraph(VGraphId virtualGraphIndex);

  // Return the name of the task which initializes/creates a poplar::Tensor in a
  // poplar::Graph. This is NOT about creating a poplar::Program.
  TaskId taskWhichCreates(TensorId) const;

  // Return the name of the task which adds code which sets the initial
  // values of poplar::Tensor to a fragment. This IS about creating a
  // poplar::Program. For Variable Tensors, this is the Copy from Stream program
  TaskId taskWhichPopulates(TensorId) const;

  // PlanningCache for matmul and conv
  poplin::PlanningCache convCache;
  poplin::matmul::PlanningCache matmulCache;

  PoplarOptions fwdConvOptions, bwdConvOptions, wuConvOptions;
  PoplarOptions fwdMmOptions, bwdMmLhsOptions, bwdMmRhsOptions;
  poplar::OptionFlags engineOptions, reportOptions;
  poplar::OptionFlags pooling_options;
  poplar::OptionFlags lstmOptions;

  PopTensors tensors;

  // Helper method to get the replication factor based on the user options
  unsigned getReplicationFactor() const;
  unsigned getAccumulationFactor() const;

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
  std::vector<ICreatorCandidatePtr> getCreatorEndpoints(
      Tensor *tensor,
      std::vector<ICreatorCandidate::OpxInAndOutIndex> pathFromInput,
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
  std::map<uint32_t, poplar::Tensor> dropoutReferenceTensors;

  poplar::Tensor getConst(poplar::Graph &graph,
                          const poplar::Type &type,
                          const std::vector<size_t> &shape,
                          double val,
                          const std::string &name);

  poplar::Tensor getScalarVariable(poplar::Graph &graph,
                                   const poplar::Type &type,
                                   const std::string &name);
  PopStreamId gradientStoreStreamId(TensorId id) const;
  PopStreamId weightLoadStreamId(TensorId id) const;

  poplar::DataStream &
  insertGradientStoreStream(TensorId, TensorInfo, poplar::Graph &);
  poplar::DataStream &
  insertWeightLoadStream(TensorId, TensorInfo, poplar::Graph &);

  const std::vector<std::pair<TensorId, TensorId>> &
  getGradAndVarStreamIds() const;
  std::vector<std::pair<TensorId, TensorId>> &getGradAndVarStreamIds();

  void connectStreamToCallback(const std::string &streamHandle,
                               std::function<void(void *)> callback);

private:
  std::unique_ptr<poplar::Graph> pGraph{nullptr};

  std::unique_ptr<poplar::Engine> pEngine{nullptr};
  std::unique_ptr<poplar::Target> pTarget{nullptr};

  std::vector<poplar::Graph> virtualGraphs;

  std::shared_ptr<DeviceInfo> deviceInfo;

  // Non-const tensors used to keep track of batch count, modulo the return
  // period
  std::map<ReturnPeriod, poplar::Tensor> batchCountingTensors;
  std::map<ReturnPeriod, poplar::Tensor> batchCountCheckingTensors;

  // Map tensors evenly across all tiles
  LinearMapper linearMapper;

  poplar::Tensor randomSeedTensor;

  PipelineInfo pInfo;
  int64_t getStashSize(VGraphId vGraphId);

  std::map<PipelineStage, VGraphId> getPipelineToVGraphIdMap() const;
  PipelineStage getMaxPipelineStage() const;

  // Task to create a poplar::Tensor from nothing, choosing
  // the correct create call (createWeights, addLinearly, etc)
  PriTask initTensorTask(Tensor *);
  PriTask initTensorByCloningTask(Op *op, TensorId srcId, TensorId dstId);
  TaskId initTensorTaskId(TensorId) const;

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

  // Task to append a Copy to poplar::Stream from poplar::Tensor
  PriTask toHostTask(Tensor *tensor,
                     poplar::program::Sequence &,
                     bool isAnchorStream) const;
  TaskId toHostTaskId(TensorId, bool isAnchorStream) const;

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
  void opTaskFunc(Op *);
  void pipelinedOpTaskFunc(Op *);
  void growOpx(Opx *, poplar::program::Sequence &);

  TaskId opTaskId(Op *) const;

  void addOpTasks(PriTasks &);

  TaskId pipelinedCopyTaskId(Op *op) const;

  void addPipelinedCopyTasks(PriTasks &);

  PriTask pipelinedCopyTask(Op *, TaskId prevTaskId);

  // The ID of the poplar::Stream host->device for poplar::Tensor
  PopStreamId h2dId(TensorId) const;
  // and for device->host
  PopStreamId d2hId(TensorId, bool isAnchorStream) const;

  bool doRearrangeOnHost(Tensor *tensor) const;

  // Hack need to for subgraph. do better
public:
  std::unique_ptr<Opx> createOpx(Op *);

  // 1-to-1 mapping between Ops and Opxs
  std::map<OpId, std::unique_ptr<Opx>> opxs;

  // Some functions useful for logging the order in which Ops are used to
  // generate poplar code / recomputed.
  //
  // The Ops in order of code generation/recompute
  const std::vector<Op *> &getMainGraphOpSeries() const;

  // index of first appearance of Op in series
  std::map<Op *, int> getMainGraphOpSeriesNums() const;

  // number of appearances of each Op. Expectation: RECOMPUTE Ops appear twice
  // and CHECKPOINT Ops appear once
  std::map<Op *, int> getMainGraphOpCounts() const;

  // A summary string of the Op series, with annotation for recomputation
  std::string getMainGraphOpString() const;

  // Returns true if using synthetic data, false if using real data
  // This will return the options.ignoreData flag
  bool useSyntheticData() const;

  bool prepareHasBeenCalled() const { return prepareHasBeenCalled_; }

private:
  std::vector<Op *> mainGraphOpRegistery;

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

  // Map of tensors to the the data streams
  std::map<TensorId, std::shared_ptr<InputDatastream>> inputStreams;
  std::map<TensorId, std::shared_ptr<OutputDatastream>> outputStreams;

  // T11630: should we combine the above with the below

  //  poplar::Streams for poplar::Tensors,
  //  1) from host to device;
  std::map<TensorId, poplar::DataStream> fromHostStreams;

  // and 2) from device to host;
  std::map<TensorId, poplar::DataStream> toHostAnchorStreams;
  std::map<TensorId, poplar::DataStream> toHostWeightStreams;

  // Streams for doing allreduce on host side
  std::map<TensorId, poplar::DataStream> toHostGradientStreams;
  std::map<TensorId, poplar::DataStream> fromHostWeightLoadStreams;

  std::vector<std::pair<TensorId, TensorId>> gradAndVarStreamIds;

  // Q: Consider replacing the d2h weight buffer with a data stream as
  // done for inputs
  std::map<TensorId, std::vector<char>> d2hWeightBuffers;

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
  void run(PopPrograms::ProgramIndex ind);

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
  // be set to `boost::none'.
  poplar::Executable getExecutable();

  // Try to save the argument executable to a file at
  // `ir().getSessionOptions().cachePath'.
  void trySaveExecutable(poplar::Executable &);

  // Try to load a poplar::Executable from a file at
  // `ir().getSessionOptions().cachePath'. If successful,
  // `this->cachedExecutable' will be set else, `this->cachedExecutable' will
  // remain set to `boost::none'.
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

  optional<poplar::Executable> cachedExecutable;
  bool usingCachedExecutable = false;

  // Option to trace the opx execution using printTensor. This can be useful in
  // determining where an exception has occurred. It is enabled by setting
  // POPART_OPX_TRACE environment variable to "1"
  bool opxTrace = false;
  poplar::Tensor opxTraceTensor;

  // This keeps track of whether there the accumulateOuterFragment  is empty
  // TODO T12001 a class which encapsulates framgments which has this attribute.
  bool outerLoopFragEmpty = true;

public:
  bool getOuterLoopFragEmpty() const { return outerLoopFragEmpty; }
};

} // namespace popx
} // namespace popart

#endif
