#ifndef GUARD_NEURALNET_DEVICE_HPP
#define GUARD_NEURALNET_DEVICE_HPP

#include <willow/names.hpp>
#include <willow/tensorinfo.hpp>

namespace willow{

class Device{

  public:
    Device(const Graph *);
    virtual ~Device();
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    // for IPUs, this will be creating the poplar::Graph,
    // poplar::Engine setting up poplar::Streams etc
    virtual void prepare() = 0;

    // write to device, from an onnx model loaded from directory,
    virtual void weightsFromHost() = 0;

    // write whatever optimizer tensors (learning rates,
    // momentum, initial momentum tensors (zero)) there are to device
    virtual void optimizerFromHost() = 0;

    // take training steps, number of steps specified in DataFlow
    // input data from address in "in"
    // output data to addrresses in "out"
    // TODO : sort out input and output locations.
    virtual void step(const std::map<TensorId, const void *> &in,
                      const std::map<TensorId, void *> &out) = 0;

    // write current model to ONNX file
    virtual void modelToHost(std::string fn) = 0;


};

}

#endif
