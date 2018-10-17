#ifndef GUARD_NEURALNET_POPDEVICE_HPP
#define GUARD_NEURALNET_POPDEVICE_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
// The protobuf generated ONNX classes
#include <poplar/tensor.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

#include <willow/device.hpp>

namespace willow {

class PopDevice : public Device {

public:
  PopDevice(const Graph *);
  virtual void prepare() override final;
};

} // namespace willow

#endif
