#ifndef GUARD_NEURALNET_DEVICE_HPP
#define GUARD_NEURALNET_DEVICE_HPP

#include <set>
#include <poponnx/names.hpp>
// MutableVoidData is defined in here:
#include <poponnx/tensordata.hpp>

namespace poponnx {

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
  virtual void optimizerFromHost()       = 0;
  virtual void infer(const IStepIO &)    = 0;
  virtual void evaluate(const IStepIO &) = 0;
  virtual void train(const IStepIO &)    = 0;
  const Ir &ir() const { return _ir; }
  virtual std::string getSummaryReport() const                         = 0;
  virtual std::string getGraphReport() const                           = 0;
  virtual std::string getExecutionReport() const                       = 0;
  virtual TensorTileMap getTensorTileMap() const                       = 0;
  virtual std::set<TensorId> getLinearlyCreatedInputTensors() const    = 0;
  virtual std::set<TensorId> getEfficientlyCreatedInputTensors() const = 0;

private:
  const Ir &_ir;
};

} // namespace poponnx

#endif
