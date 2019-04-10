#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>

#include <poponnx/device.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/popx/enigma.hpp>
#include <poponnx/popx/poplaroptionsx.hpp>
#include <poponnx/popx/subgraphoutlinex.hpp>
#include <poponnx/pritask.hpp>

namespace poponnx {
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
    WEIGHTSFROMHOST = 0,
    OPTIMIZERFROMHOST,
    PROGRAM,
    WEIGHTSTOHOST,
    SETRANDOMSEED,
    N // The number of program fragments
  };

  // Program fragments are not necessarily complete program that can be given to
  // a poplar engine.
  poplar::program::Sequence &weightsFromHostFragment();
  poplar::program::Sequence &optimizerFromHostFragment();
  poplar::program::Sequence &setRandomSeedFragment();
  poplar::program::Sequence &programFragment();
  poplar::program::Sequence &weightsToHostFragment();

  // A list of programs that can be run by the Poplar engine.
  std::vector<poplar::program::Program> progs();

  poplar::program::Sequence &programFragment(PopPrograms::ProgramFragmentIndex);

private:
  // Specify how many times to loop the 'repeatable' programs
  // (infer, eval, train)
  int repeatCount;

  static constexpr int seqs_size = static_cast<int>(ProgramFragmentIndex::N);
  std::array<poplar::program::Sequence, seqs_size> seqs;

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
  OpxInAndOutIndex(Opx *opx_, InIndex inIndex_, OutIndex outIndex_)
      : opx(opx_), inIndex(inIndex_), outIndex(outIndex_) {}
  OpxInAndOutIndex() = default;

  Opx *opx;
  InIndex inIndex;
  OutIndex outIndex;
};

// A bundle class to represent candidate Opxs
// for allocating an input tensor
class InputCreatorCandidate {
public:
  InputCreatorCandidate(int, Opx *, std::vector<OpxInAndOutIndex>);
  InputCreatorCandidate() = default;
  int index;
  Opx *opx;

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

class Devicex : public poponnx::Device {

public:
  Devicex(const Ir &, std::shared_ptr<DeviceInfo> deviceInfo);
  void prepare() final;
  void weightsFromHost() final;
  void optimizerFromHost() final;

  void run(const IStepIO &) final;
  void weightsToHost() final;
  void weightsToHost(const std::map<TensorId, MutableVoidData> &) final;
  void readWeights(const IWeightsIO &weights) final;
  void writeWeights(const IWeightsIO &weights) final;

  virtual std::string getSummaryReport() const override final;
  virtual std::string
  getGraphReport(bool use_cbor = false) const override final;
  virtual std::string
  getExecutionReport(bool use_cbor = false) const override final;
  virtual TensorTileMap getTensorTileMap() const override final;

  // Return stored input tensors based on how they are allocated
  virtual std::set<TensorId>
  getLinearlyCreatedInputTensors() const override final;
  virtual std::set<TensorId>
  getEfficientlyCreatedInputTensors() const override final;

  PopPrograms progs;

  Opx *getOpx(OpId);

  // Get the root graph
  poplar::Graph &rootGraph();
  const poplar::Graph &rootGraph() const;

  poplar::Graph &masterGraph();
  poplar::Graph &graph(int64_t virtualGraphIndex);

  // return the name of the task which creates a poplar::Tensor
  // This function is mostly string manipulation
  TaskId taskWhichCreates(TensorId) const;

  // enigma has a PlanningCache for matmul and conv
  poplin::PlanningCache convCache;
  poplin::matmul::PlanningCache matmulCache;

  PoplarOptions fwdConvOptions, bwdConvOptions, wuConvOptions;
  PoplarOptions fwdMmOptions, bwdMmLhsOptions, bwdMmRhsOptions;
  poplar::OptionFlags engineOptions, reportOptions;
  poplar::OptionFlags pooling_options;
  poplar::OptionFlags lstmOptions;

  PopTensors tensors;

  poplar::Tensor getConst(const poplar::Type &type,
                          const std::vector<size_t> &shape,
                          double val);

  // Helper method to get the replication factor based on the user options
  unsigned getReplicationFactor() const;

private:
  // The root graph. Operations that span the boundaries between
  // replicated subgraphs (e.g. all-reduce of weight deltas) should be added
  // here
  std::unique_ptr<poplar::Graph> pRootGraph{nullptr};

  // Operations that are not mapped to a specific IPU should be added to
  // this graph. This will be a replicated graph if the options specify a
  // replication factor greater than one.
  std::unique_ptr<poplar::Graph> pMasterGraph{nullptr};

  std::unique_ptr<poplar::Engine> pEngine{nullptr};
  std::unique_ptr<poplar::Target> pTarget{nullptr};

  std::vector<poplar::Graph> virtualGraphs;

  std::shared_ptr<DeviceInfo> deviceInfo;

  // Non-const tensors used to keep track of batch count, modulo the return
  // period
  std::map<ReturnPeriod, poplar::Tensor> batchCountingTensors;
  std::map<ReturnPeriod, poplar::Tensor> batchCountCheckingTensors;

  // A forward search of graph:
  //   - from inputs of the graph
  //   - to Opxs with optimized poplar calls to create the tensor,
  //     or to Opxs that destroy layout information of the input
  //     tensor on the output
  //   - traversing through Opxs that cannot create the tenosr
  //     themselves, but preserve layout information from input
  //     to output tensor
  //   - tracking the route taken through the graph to the endpoints
  // Using the defualt arguments will return only creator candidates,
  // with each candidate's path containing only Opxs that need to be
  // 'unwound' to correctly lay out the input tensor
  std::vector<InputCreatorCandidate>
  getCreatorEndpoints(Tensor *tensor,
                      std::vector<OpxInAndOutIndex> pathFromInput,
                      bool excludeEndpointsFromPath = true,
                      bool includeDeadends          = false);

  // Task to create a poplar::Tensor from nothing, choosing
  // the correct create call (createWeights, addLinearly, etc)
  PriTask initTensorTask(Tensor *);
  TaskId initTensorTaskId(TensorId) const;

  PriTask initRandomSeed();
  void connectRandomSeedStream();

  PriTask setInitTensorValTask(Tensor *);
  TaskId setInitTensorValTaskId(TensorId);

  // Task to create a poplar::Stream to write to poplar::Tensor
  // C++ Note: if a lambda function which modifies `this' is created
  // it must be const w.r.t this, even if it not run
  PriTask streamFromHostTask(Tensor *);
  TaskId streamFromHostTaskId(TensorId) const;

  // Task to create a poplar::Stream to write from poplar::Tensor
  PriTask streamToHostTask(Tensor *);
  TaskId streamToHostTaskId(TensorId) const;

  // Task to append a Copy from poplar::Stream to poplar::Tensor
  PriTask fromHostTask(Tensor *tensor, poplar::program::Sequence &) const;
  TaskId fromHostTaskId(TensorId) const;

  // Task to append a Copy to poplar::Stream from poplar::Tensor
  PriTask toHostTask(Tensor *tensor, poplar::program::Sequence &) const;
  TaskId toHostTaskId(TensorId) const;

  // Task to create poplar::Tensors from nothing, specifically for
  // use in keeping track of the batch count
  PriTask initBatchCounterTensorsTask();
  TaskId initBatchCounterTensorsTaskId() const;

  // Task to add a program to increment and check the batch count
  PriTask updateBatchCoutTask(poplar::program::Sequence &sq);
  TaskId updateBatchCoutTaskId() const;

  // Task to append a Copy to poplar::Stream from poplar::Tensor every
  // N batches
  PriTask toHostEveryNBatchesTask(Tensor *tensor,
                                  ReturnPeriod N,
                                  poplar::program::Sequence &);

  PriTask opTask(Op *, double priority, TaskId prevOpTaskId);
  TaskId opTaskId(Op *) const;

  // The ID of the poplar::Stream host->device for poplar::Tensor
  PopStreamId h2dId(TensorId) const;
  // and for device->host
  PopStreamId d2hId(TensorId) const;

  // Hack need to for subgraph. do better
public:
  std::unique_ptr<Opx> createOpx(Op *);

  // 1-to-1 mapping between Ops and Opxs
  std::map<OpId, std::unique_ptr<Opx>> opxs;

private:
  // the poplar::Streams for poplar::Tensors,
  // from host to device:
  std::map<TensorId, poplar::DataStream> fromHostStreams;
  // and from device to host:
  std::map<TensorId, poplar::DataStream> toHostStreams;

  std::map<TensorId, std::vector<char>> h2dBuffers;
  std::map<TensorId, std::vector<char>> d2hBuffers;

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

  poplar::program::Sequence &programFragment();

  // Returns true if using synthetic data, false if using real data
  // This will return the options.ignoreData flag
  bool useSyntheticData();

  template <typename T> void setInitVal(Tensor *tensor);
  void setInitValHalf(Tensor *tensor);

  // Store input tensors based on how they are allocated
  std::set<TensorId> linearlyCreatedInputTensors;
  std::set<TensorId> efficientlyCreatedInputTensors;

  // The subgraph outliner
  SubgraphOutlinex outline;
};

} // namespace popx
} // namespace poponnx

#endif
