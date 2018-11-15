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
#include <poponnx/popx/enigma.hpp>
#include <poponnx/pritask.hpp>

namespace willow {
namespace popx {

using PopStreamId = std::string;

class Opx;

class PopPrograms {

public:
  enum ProgramIndex {
    WEIGHTSFROMHOST = 0,
    OPTIMIZERFROMHOST,
    INFER,
    EVALUATE,
    TRAIN,
    WEIGHTSTOHOST,
    N // The number of programs
  };

  // Program fragments are not necessarily complete program that can be given to
  // a poplar engine.
  poplar::program::Sequence &weightsFromHostFragment();
  poplar::program::Sequence &optimizerFromHostFragment();
  poplar::program::Sequence &forwardFragment();
  poplar::program::Sequence &lossFragment();
  poplar::program::Sequence &backwardFragment();
  poplar::program::Sequence &weightsToHostFragment();

  std::vector<poplar::program::Program> progs();

private:
  enum class ProgramFragmentIndex {
    WEIGHTSFROMHOST = 0,
    OPTIMIZERFROMHOST,
    FORWARD,
    LOSS,
    BACKWARD,
    WEIGHTSTOHOST,
    N // The number of program fragments
  };

  static constexpr int seqs_size = static_cast<int>(ProgramFragmentIndex::N);
  std::array<poplar::program::Sequence, seqs_size> seqs;

  poplar::program::Sequence weightsFromHost();
  poplar::program::Sequence optimizerFromHost();
  poplar::program::Sequence infer();
  poplar::program::Sequence evaluate();
  poplar::program::Sequence train();
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
  // with the corresponding willow::Tensor's
  const Ir &ir;
};

class Devicex : public willow::Device {

public:
  Devicex(const Ir &);
  void prepare() final;
  void weightsFromHost() final;
  void optimizerFromHost() final;
  void infer(const StepIO &) final;
  void evaluate(const StepIO &) final;
  void train(const StepIO &) final;
  void weightsToHost(const std::map<TensorId, MutableVoidData> &) final;

  PopPrograms progs;
  Opx *getOpx(OpId);
  poplar::Graph &graph();

  // return the name of the task which creates a poplar::Tensor
  // This function is mostly string manipulation
  TaskId taskWhichCreates(TensorId) const;

  // enigma has a PlanningCache for matmul and conv
  poplin::PlanningCache convCache;
  poplin::matmul::PlanningCache matmulCache;

  // completed in Devicex constructor.

  enigma::ConvOptions fwdConvOptions, bwdConvOptions, wuConvOptions;
  poplar::OptionFlags fwdMmOptions, bwdMmLhsOptions, bwdMmRhsOptions;
  poplar::OptionFlags engineOptions;

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
};

} // namespace popx
} // namespace willow

#endif
