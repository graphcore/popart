#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#include <boost/optional.hpp>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>

#include <poponnx/devicemanager.hpp>
#include <poponnx/popx/enigma.hpp>
#include <poponnx/popx/linearmapper.hpp>
#include <poponnx/popx/poplaroptionsx.hpp>
#include <poponnx/pritask.hpp>

#include <set>
#include <poponnx/names.hpp>
// MutableVoidData is defined in here:
#include <poponnx/tensordata.hpp>

using boost::optional;

namespace poponnx {

enum class ScheduledPreLoss;

namespace popx {

using PopStreamId = std::string;

class Opx;
class GraphCachex;

class PopPrograms {

public:
  // We may want to run some programs multiple times without having
  // to communicate with the host to call the 'run'. By supplying a
  // count, we can loop a repeatable program inside a Poplar repeat
  // program
  PopPrograms(const int repeatCount);

  enum ProgramIndex {
    WEIGHTSFROMHOST = 0,
    OPTIMIZERFROMHOST,
    PROGRAM,
    WEIGHTSTOHOST,
    N // The number of programs
  };

  // Order of these enums is used for scheduling
  enum class ProgramFragmentIndex {
    STREAMWEIGHTSFROMHOST = 0,
    STREAMOPTIMIZERFROMHOST,
    INIT,
    PREFORWARD,
    FORWARD,
    BACKWARD,
    WEIGHTSTOHOST,
    TOHOSTFINALCOPY,
    SETRANDOMSEED,
    SETRANDOMDROPOUTSEED,
    N // The number of program fragments
  };

  // Program fragments are not necessarily complete program that can be given to
  // a poplar engine.
  poplar::program::Sequence &streamWeightsFromHostFragment();
  poplar::program::Sequence &streamOptimizerFromHostFragment();
  poplar::program::Sequence &setRandomSeedFragment();
  poplar::program::Sequence &setRandomDropoutSeedFragment();
  poplar::program::Sequence &toHostFinalCopyFragment();
  poplar::program::Sequence &initFragment();
  poplar::program::Sequence &preForwardFragment();
  poplar::program::Sequence &forwardFragment();
  poplar::program::Sequence &backwardFragment();
  poplar::program::Sequence &weightsToHostFragment();
  // If ScheduledPreLoss::Yes, then return forwardFragment(), else return
  // backwardFragment()
  poplar::program::Sequence &forwardOrBackwardFragment(ScheduledPreLoss);

  // A list of programs that can be run by the Poplar engine.
  std::vector<poplar::program::Program> progs();

  poplar::program::Sequence &programFragment(PopPrograms::ProgramFragmentIndex);

  // Sub-graph program fragments, getters and setters
  poplar::program::Sequence &scopeFragment(const Graph &);
  bool containsFragment(const Graph &) const;
  void createFragment(const Graph &);

  // Recompute program fragments, get and (implicitly) create. There is a unique
  // fragment for each recomputed Op
  poplar::program::Sequence &recomputeFragment(OpId id);

  bool hasBeenRecomputed(OpId) const;
  void recordRecomputed(OpId id);

private:
  // Specify how many times to loop the 'repeatable' program
  int repeatCount;

  static constexpr int seqs_size = static_cast<int>(ProgramFragmentIndex::N);
  std::array<poplar::program::Sequence, seqs_size> seqs;

  // The sub-graph program fragments will be stored here
  std::unordered_map<std::string, poplar::program::Sequence> scopeSeqs;

  // The recompute program fragments will be stored here
  std::map<OpId, poplar::program::Sequence> recomputeSeqs;

  std::set<OpId> beenRecomputed;

  poplar::program::Sequence weightsFromHost();
  poplar::program::Sequence optimizerFromHost();
  poplar::program::Sequence program();
  poplar::program::Sequence weightsToHost();
};

poplar::Type popType(const TensorInfo &);
poplar::Type popType(DataType);

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

// A bundle class to represent candidate Opxs
// for allocating an input tensor
class InputCreatorCandidate {
public:
  InputCreatorCandidate(int, const Opx *, std::vector<OpxInAndOutIndex>);
  InputCreatorCandidate() = default;
  int index;
  const Opx *opx;

  std::vector<OpxInAndOutIndex> getPathFromInput();

private:
  std::vector<OpxInAndOutIndex> pathFromInput;
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

  void run(const IStepIO &);

  // device -> host stream
  void weightsToHost();
  // device ->host stream -> specified host addresses
  void weightsToHost(const std::map<TensorId, MutableVoidData> &);

  // TODO T8229 : change these names to disambiguate
  // the source and destination
  // (is this writing to or from the device?)
  void readWeights(const IWeightsIO &weights);
  void writeWeights(const IWeightsIO &weights);

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

  // Get the root graph
  poplar::Graph &rootGraph();
  const poplar::Graph &rootGraph() const;

  // T9669 replace masterGraph with replicatedGraph
  poplar::Graph &masterGraph();

  poplar::Graph &replicatedGraph();
  const poplar::Graph &replicatedGraph() const;

  poplar::Graph &graph(int64_t virtualGraphIndex);

  // Return the name of the task which initializes/creates a poplar::Tensor in a
  // poplar::Graph. This is NOT about creating a poplar::Program.
  TaskId taskWhichCreates(TensorId) const;

  // Return the name of the task which adds code which sets the final
  // values of poplar::Tensor to a fragment. This IS about creating a
  // poplar::Program.
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

  bool containsFragment(const Graph &scope) const;
  void createFragment(const Graph &);

  // A forward search of graph:
  //   - from inputs of the graph
  //   - to Opxs with optimized poplar calls to create the tensor,
  //     or to Opxs that destroy layout information of the input
  //     tensor on the output
  //   - traversing through Opxs that cannot create the tensor
  //     themselves, but preserve layout information from input
  //     to output tensor
  //   - tracking the route taken through the graph to the endpoints
  // Using the default arguments will return only creator candidates,
  // with each candidate's path containing only Opxs that need to be
  // 'unwound' to correctly lay out the input tensor
  std::vector<InputCreatorCandidate>
  getCreatorEndpoints(Tensor *tensor,
                      std::vector<OpxInAndOutIndex> pathFromInput,
                      bool excludeEndpointsFromPath = true,
                      bool includeDeadends          = false) const;

  // Get a single creator candidate for creating a tensor
  // Will throw an error if multiple candidates that do not agree are found
  optional<InputCreatorCandidate> getTensorCreator(Tensor *tensor) const;

  bool isEngineLoaded() const;
  void setEngineIsLoaded(bool isLoaded);

  bool isDropoutRandomSeedRequired() const;
  void setDropoutRandomSeedIsRequired(bool isRequired);
  std::string dropoutRandomSeedTensorId() const;
  const poplar::Tensor *getDropoutRandomSeed() const;
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

private:
  // The root graph. Operations that span the boundaries between
  // replicated subgraphs (e.g. all-reduce of weight deltas) should be added
  // here
  std::unique_ptr<poplar::Graph> pRootGraph{nullptr};

  // Operations that are not mapped to a specific IPU should be added to
  // this graph. This will be a replicated graph if the options specify a
  // replication factor greater than one.
  std::unique_ptr<poplar::Graph> pReplicatedGraph{nullptr};

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

  // A random seed tensor is needed to get repeatable random
  // masks for corresponding fwd and bwd dropout ops.
  // Design decision: a separate seed tensor to the one used to
  // initialise the random hardware so that manipulation of this
  // tensor for some other reason between fwd and bwd dropout
  // layers doensn't break dropout's functionality
  bool requiresDropoutRandomSeed = false;
  poplar::Tensor dropoutRandomSeed;

  // Task to create a poplar::Tensor from nothing, choosing
  // the correct create call (createWeights, addLinearly, etc)
  PriTask initTensorTask(Tensor *);
  PriTask initTensorByCloningTask(Op *op, TensorId srcId, TensorId dstId);
  TaskId initTensorTaskId(TensorId) const;

  PriTask initRandomSeed();
  void connectRandomSeedStream();
  PriTask initDropoutRandomSeed();
  TaskId initDropoutRandomSeedId() const;

  PriTask setInitTensorValTask(Tensor *);
  TaskId setInitTensorValTaskId(TensorId) const;

  // Task to create a poplar::Stream to write to poplar::Tensor
  // C++ Note: if a lambda function which modifies `this' is created
  // it must be const w.r.t this, even if it not run
  PriTask streamFromHostTask(Tensor *);
  TaskId streamFromHostTaskId(TensorId) const;

  // Task to create a poplar::Stream to write from poplar::Tensor
  PriTask streamToHostTask(Tensor *);
  TaskId streamToHostTaskId(TensorId) const;

  // Task to append a Copy from poplar::Stream to poplar::Tensor
  PriTask fromHostTask(Tensor *tensor,
                       poplar::program::Sequence &streamSq) const;
  TaskId fromHostTaskId(TensorId) const;

  // Task to append a Copy to poplar::Stream from poplar::Tensor
  PriTask toHostTask(Tensor *tensor, poplar::program::Sequence &) const;
  TaskId toHostTaskId(TensorId) const;

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

  PriTask incrementDropoutRandomSeedTask();

  PriTask opTask(Op *, double priority, TaskId prevOpTaskId);

  TaskId opTaskId(Op *) const;

  void addOpTasks(PriTasks &);

  // The ID of the poplar::Stream host->device for poplar::Tensor
  PopStreamId h2dId(TensorId) const;
  // and for device->host
  PopStreamId d2hId(TensorId) const;

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

private:
  std::vector<Op *> mainGraphOpRegistery;

  //  poplar::Streams for poplar::Tensors,
  //  1) from host to device;
  std::map<TensorId, poplar::DataStream> fromHostStreams;
  // and 2) from device to host;
  std::map<TensorId, poplar::DataStream> toHostStreams;

  std::map<TensorId, std::vector<char>> h2dBuffers;
  std::map<TensorId, std::vector<char>> d2hBuffers;

  // Wrapper for calls to poplar Engine API calls: loading
  // engine onto the poplar device and connecting streams.
  // Must be called before running a poplar program with a
  // call to this Devicex's engine.
  void loadEngineAndConnectStreams();

  // Is this Devicex's engine the last to have been loaded onto
  // deviceInfo's device?
  // Becomes true once loadEngineAndConnectStreams() is called.
  // Becomes 'false' if another engine has been loaded after
  // loadEngineAndConnectStreams() has been called. This is
  // different to 'prepareHasBeenCalled', which, once true,
  // is always true
  bool engineIsLoaded = false;

  // Wrapper function that checks the calling devicex was the
  // last to have loaded its engine to deviceInfo's device
  void run(PopPrograms::ProgramIndex ind);

  // copy a step tensor from user provided src, to allocated memory dst
  // input parameters are,
  // dst     : destination of copy, this is the host end of a poplar::Stream
  // src     : source of the copy
  // dstInfo : the info for dst
  // srcInfo : user provided info for src. Both TensorInfos are required
  //           so that we can verify dst and src are the same size
  void hostToHostStream(void *dst,
                        const void *src,
                        const TensorInfo &dstInfo,
                        const TensorInfo &srcInfo,
                        TensorId id);

  void hostStreamToHost(const MutableVoidData &mv_data, TensorId id);

  // Call hostToHostStream on all the Tensors in pir->dataStreamTensors()
  void anchorsHostToHostStreams(const IStepIO &stepio);

  // Call hostStreamToHost in all the Tensors in pir->dataFlow.anchors()
  void anchorsHostFromHostStreams(const IStepIO &stepio);

  // Returns true if using synthetic data, false if using real data
  // This will return the options.ignoreData flag
  bool useSyntheticData() const;

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
  std::string getPoponnxCachePath();

  void setFloatingPointBehaviour(poplar::Graph &graph);
  void setStochasticRoundingBehaviour(poplar::Graph &graph);

  void doProfileChecks() const;

  // Store input tensors based on how they are allocated
  std::set<TensorId> linearlyCreatedInputTensors;
  std::set<TensorId> efficientlyCreatedInputTensors;

  bool prepareHasBeenCalled;

  optional<poplar::Executable> cachedExecutable;
  bool usingCachedExecutable = false;
};

} // namespace popx
} // namespace poponnx

#endif
