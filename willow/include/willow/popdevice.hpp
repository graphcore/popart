#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#include <willow/device.hpp>

namespace willow {

class PopDevice : Device {

public:
  PopDevice(const Graph *);
  virtual ~PopDevice() = default;

  // this will be creating the poplar::Graph,
  // poplar::Engine setting up poplar::Streams etc
  virtual void prepare() = 0;

  virtual void weightsFromHost() = 0;

  // write whatever optimizer tensors (learning rates,
  // momentum, initial momentum tensors (zero)) there are to device
  virtual void optimizerFromHost() = 0;

  // For Poplar, this will involve reading and writing
  // Poplar::Stream <--> these addresses.
  virtual void step(const std::map<TensorId, const void *> &in,
                    const std::map<TensorId, void *> &out) = 0;

  // write current model to ONNX file
  virtual void modelToHost(std::string fn) = 0;
};

} // namespace willow

#endif
