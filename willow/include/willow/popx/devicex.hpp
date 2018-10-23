#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/Convolution.hpp>
#include <poputil/TileMapping.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

#include <willow/device.hpp>
#include <willow/popx/enigma.hpp>
#include <willow/pritask.hpp>

namespace willow {
namespace popx {

using PopStreamId = std::string;

class Opx;

class PopPrograms {

public:
  enum ProgramIndex {
    WEIGHTSFROMHOST = 0,
    OPTIMIZERFROMHOST,
    STEP,
    WEIGHTSTOHOST,
    N // The number of programs
  };

  poplar::program::Sequence &weightsFromHost();
  poplar::program::Sequence &optimizerFromHost();
  poplar::program::Sequence &step();
  poplar::program::Sequence &weightsToHost();
  std::vector<poplar::program::Program> progs();

private:
  std::array<poplar::program::Sequence, ProgramIndex::N> seqs;
};

poplar::Type popType(const TensorInfo &);

// A bundle class for an int and an Opx.
class OpxAndInIndex {
public:
  OpxAndInIndex(int, Opx *);
  OpxAndInIndex() = default;
  int index;
  Opx *opx;
};

class Devicex : public willow::Device {

public:
  Devicex(const Ir *);
  virtual void prepare() override final;
  void weightsFromHost() override final;
  void optimizerFromHost() override final;
  Opx *getOpx(OpId);
  poplar::Graph &graph();

  // enigma has a PlanningCache for matmul and conv
  poplin::PlanningCache convCache;
  poplin::PlanningCache matmulCache;

  // completed in Devicex constructor.
  enigma::ConvOptions fwdConvOptions, bwdConvOptions, wuConvOptions;
  poplar::OptionFlags engineOptions;

  // return the name of the task which creates a poplar::Tensor
  // This function is pure string manipulation
  TaskId taskWhichCreates(TensorId);

private:
  std::unique_ptr<poplar::Graph> pGraph{nullptr};
  std::unique_ptr<poplar::Engine> pEngine{nullptr};
  std::unique_ptr<poplar::Target> pTarget{nullptr};
  poplar::Device popDevice;

  PopPrograms progs;

  // Task to create a poplar::Tensor, choosing
  // the correct create call (createWeights, addLinearly, etc)
  PriTask popTensorTask(Tensor *tensor);
  TaskId popTensorTaskId(TensorId);

  // Task to create a poplar::Stream to write to poplar::Tensor
  PriTask streamFromHostTask(Tensor *tensor);
  TaskId streamFromHostTaskId(TensorId);

  // Task to append a Copy from poplar::Stream to poplar::Tensor
  PriTask fromHostTask(Tensor *tensor, poplar::program::Sequence &);
  TaskId fromHostTaskId(TensorId);

  // The ID of the poplar::Stream host->device for poplar::Tensor
  PopStreamId h2dId(TensorId);

  std::unique_ptr<Opx> createOpx(Op *);

  // 1-to-1 mapping between Ops and Opxs
  std::map<OpId, std::unique_ptr<Opx>> opxs;
  std::map<TensorId, poplar::Tensor> popTensors;

  // the poplar::Streams for poplar::Tensors,
  // from host to device:
  std::map<TensorId, poplar::DataStream> fromHostStreams;
  // and from device to host:
  std::map<TensorId, poplar::DataStream> toHostStreams;
};

} // namespace popx
} // namespace willow

#endif
