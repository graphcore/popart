#ifndef GUARD_NEURALNET_NET_HPP
#define GUARD_NEURALNET_NET_HPP

#include <poponnx/names.hpp>

namespace willow {

class Net {
public:
  Net(std::string fnOnnxModel,
      const EarlyInfo &,
      const DataFlow &,
      const std::vector<Loss *> &,
      const Optimizer *,
      const std::vector<std::string> &cTens,
      std::string logdir_,
      const std::vector<std::string> &patternNames);

  ~Net();

  // Update the optimizer. Note that the optimizer passed in
  // must be compatible with that passed to the constructor.
  // For example, you cannot update to an Optimizer which uses
  // momentum here, if the Optimizer passed to the constructor
  // did not have momentum. Reason: The Ir would need to change
  // to incorporate momentum, but the Ir is frozen once
  // constructed. NB: Must call optimizerToDevice for this update
  // to take effect on the device.
  void updateOptimizer(const Optimizer *);

  // What device to use (TODO T5105) ? IPU Model, IPU, IPU-CPU,
  // our own CPU backend (TODO T5103), choose automatically,
  // several IPUs ?
  void setDevice(std::string x);

  // for IPUs, this will be creating the poplar::Graph,
  // poplar::Engine, and setting up poplar::Streams.
  void prepareDevice();

  // write to device, from an ONNX model loaded from directory
  // Currently, the weights are taken from the onnx Model passed to the
  // constructor, but this should be relaxed so that the weights can
  // come from any Model: (TODO T5207)
  void weightsFromHost();

  // write whatever optimizer tensors (learning rates,
  // momentum, initial momentum tensors (zero)) there are to device
  void optimizerFromHost();

  // take 1 training step (i.e. process nBatchesPerStep batches, where
  // nBatchesPerStep is specified in the DataFlow object passed into
  // the constructor).
  // input data  : from address in stepIO.in
  // output data : to addresses in stepIO.out
  // For Poplar, this involves reading and writing between
  // poplar::Stream host addresses and the addresses specified in stepIO,
  // and then between the Stream host addresses and the IPU tensors.
  void step(const StepIO &stepIO);

  // write current model to ONNX file
  void modelToHost(std::string fn);

  // get the TensorInfo on a Tensor
  TensorInfo getInfo(TensorId) const;

private:
  // abstraction of the computation, the Ir is where
  // all the compute graph optimisations, backwards pass construction,
  // recomputation growing etc. happens.
  std::unique_ptr<Ir> pir_;
  // Implementation of the computation, for IPU backend this is
  // where calls to poplar are made.
  std::unique_ptr<Device> device_;
};
} // namespace willow

#endif
