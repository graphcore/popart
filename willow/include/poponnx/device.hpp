#ifndef GUARD_NEURALNET_DEVICE_HPP
#define GUARD_NEURALNET_DEVICE_HPP

#include <poponnx/names.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensorinfo.hpp>

namespace willow {

class Device {

public:
  Device(const Ir &g) : _ir(g) {}
  virtual ~Device()      = default;
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  virtual void prepare()            = 0;
  virtual void weightsFromHost()    = 0;
  virtual void weightsToHost(const std::map<TensorId, MutableVoidData> &) = 0;
  // write optimizer-specific tensors (learning rates, etc.) to Device
  virtual void optimizerFromHost()  = 0;
  virtual void step(const StepIO &) = 0;
  const Ir &ir() const { return _ir; }

private:
  const Ir &_ir;
};

} // namespace willow

#endif
