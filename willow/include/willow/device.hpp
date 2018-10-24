#ifndef GUARD_NEURALNET_DEVICE_HPP
#define GUARD_NEURALNET_DEVICE_HPP

#include <willow/names.hpp>
#include <willow/tensorinfo.hpp>

namespace willow {

class Device {

public:
  Device(const Ir *);
  virtual ~Device();
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  virtual void prepare()            = 0;
  virtual void weightsFromHost()    = 0;
  // write optimizer-specific tensors (learning rates, etc.) to Device
  virtual void optimizerFromHost()  = 0;
  virtual void step(const StepIO &) = 0;

protected:
  const Ir *pir;
};

} // namespace willow

#endif
