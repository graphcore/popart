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
#include <poponnx/popx/convoptionsx.hpp>
#include <poponnx/popx/enigma.hpp>
#include <poponnx/popx/graphcachex.hpp>
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
    INFER,
    EVALUATE,
    TRAIN,
    WEIGHTSTOHOST,
    N // The number of programs
  };

  // Order of these enums is used for scheduling
  enum class ProgramFragmentIndex {
    WEIGHTSFROMHOST = 0,
    OPTIMIZERFROMHOST,
    FORWARD,
    LOSS,
    BACKWARD,
    WEIGHTSTOHOST,
    N // The number of program fragments
  };

  // Program fragments are not necessarily complete program that can be given to
  // a poplar engine.
  poplar::program::Sequence &weightsFromHostFragment();
  poplar::program::Sequence &optimizerFromHostFragment();
  poplar::program::Sequence &forwardFragment();
  poplar::program::Sequence &lossFragment();
  poplar::program::Sequence &backwardFragment();
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
  poplar::program::Repeat infer();
  poplar::program::Repeat evaluate();
  poplar::program::Repeat train();
  poplar::program::Sequence weightsToHost();
};

poplar::Type popType(const TensorInfo &);
poplar::Type popType(DataType);

// A bundle class for an int and an Opx.
class OpxAndInIndex {
public:
  OpxAndInIndex(int, Opx *);
  OpxAndInIndex() = default;
  int index;
  Opx *opx;
};

class PopTensors {
public:
  PopTensors(const Ir &);
  void insert(TensorId, const poplar::Tensor &);
  const poplar::Tensor &get(TensorId) const;
  bool contains(TensorId) const;

private:
  std::map<TensorId, poplar::Tensor> tensors_;
  // This Ir is used to compare the shape
  // of a poplar::Tensor added with `insert',
  // with the corresponding poponnx::Tensor's
  const Ir &ir;
};

class Devicex : public poponnx::Device {

public:
  Devicex(const Ir &, DeviceInfo &deviceInfo);
  void prepare() final;
  void weightsFromHost() final;
  void optimizerFromHost() final;
  void infer(const StepIO &) final;
  void evaluate(const StepIO &) final;
  void train(const StepIO &) final;
  void weightsToHost(const std::map<TensorId, MutableVoidData> &) final;

  virtual std::string getSummaryReport() const override final;
  virtual std::string getGraphReport() const override final;
  virtual std::string getExecutionReport() const override final;

  PopPrograms progs;

  Opx *getOpx(OpId);
  poplar::Graph &graph();

  // return the name of the task which creates a poplar::Tensor
  // This function is mostly string manipulation
  TaskId taskWhichCreates(TensorId) const;

  // enigma has a PlanningCache for matmul and conv
  poplin::PlanningCache convCache;
  poplin::matmul::PlanningCache matmulCache;

  // Poplar level graph caching
  GraphCachex graphCache;

  ConvOptions fwdConvOptions, bwdConvOptions, wuConvOptions;
  poplar::OptionFlags fwdMmOptions, bwdMmLhsOptions, bwdMmRhsOptions;
  poplar::OptionFlags engineOptions, reportOptions;

  PopTensors tensors;

  // If a tensor matching the input parameters already exists,
  // just return it. otherwise create and return.
  const poplar::Tensor &getConst(const poplar::Type &type,
                                 const std::vector<size_t> &shape,
                                 double val);

private:
  // unique identifier based on  the 3 input parameters
  std::string getConstTensorKey(const poplar::Type &,
                                const std::vector<size_t> &shape,
                                double val) const;

  std::unique_ptr<poplar::Graph> pGraph{nullptr};
  std::unique_ptr<poplar::Engine> pEngine{nullptr};
  std::unique_ptr<poplar::Target> pTarget{nullptr};
  poplar::Device popDevice;

  std::map<std::string, poplar::Tensor> constTensors;

  // Variable tensors used keep track of batch Id
  std::map<int, poplar::Tensor> batchCountingTensors;
  std::map<int, poplar::Tensor> batchCountCheckingTensors;

  // Task to create a poplar::Tensor from nothing, choosing
  // the correct create call (createWeights, addLinearly, etc)
  PriTask initTensorTask(Tensor *tensor);
  TaskId initTensorTaskId(TensorId) const;

  // Task to create a poplar::Stream to write to poplar::Tensor
  // C++ Note: if a lambda function which modifies `this' is created
  // it must be const w.r.t this, even if it not run
  PriTask streamFromHostTask(Tensor *tensor);
  TaskId streamFromHostTaskId(TensorId) const;

  // Task to create a poplar::Stream to write from poplar::Tensor
  PriTask streamToHostTask(Tensor *tensor);
  TaskId streamToHostTaskId(TensorId) const;

  // Task to append a Copy from poplar::Stream to poplar::Tensor
  PriTask fromHostTask(Tensor *tensor, poplar::program::Sequence &) const;
  TaskId fromHostTaskId(TensorId) const;

  // Task to append a Copy to poplar::Stream from poplar::Tensor
  PriTask toHostTask(Tensor *tensor, poplar::program::Sequence &) const;
  TaskId toHostTaskId(TensorId) const;

  // Task to create poplar::Tensors from nothing, specifically for
  // use in keeping track of the batch count
  PriTask initBatchCounterTensorsTask(poplar::program::Sequence &sq);
  TaskId initBatchCounterTensorsTaskId() const;

  // Task to add a program to increment and check the batch count
  PriTask updateBatchCoutTask(poplar::program::Sequence &sq);
  TaskId updateBatchCoutTaskId() const;

  // Task to append a Copy to poplar::Stream from poplar::Tensor every
  // N batches
  PriTask
  toHostEveryNBatchesTask(Tensor *tensor, int N, poplar::program::Sequence &);

  PriTask opTask(Op *, double priority);
  TaskId opTaskId(Op *) const;

  // The ID of the poplar::Stream host->device for poplar::Tensor
  PopStreamId h2dId(TensorId) const;
  // and for device->host
  PopStreamId d2hId(TensorId) const;

  std::unique_ptr<Opx> createOpx(Op *);

  // 1-to-1 mapping between Ops and Opxs
  std::map<OpId, std::unique_ptr<Opx>> opxs;

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
  void anchorsHostToHostStreams(const StepIO &stepio);

  // Call hostStreamToHost in all the Tensors in pir->dataFlow.anchors()
  void anchorsHostFromHostStreams(const StepIO &stepio);

  // Helper function to find the program fragment index an op/tensor belongs in
  static PopPrograms::ProgramFragmentIndex programFragmentIndex(Vertex *vertex);

  // Helper function to find the program fragment an op/tensor belongs in
  poplar::program::Sequence &programFragment(Vertex *);
};

} // namespace popx
} // namespace poponnx

#endif
