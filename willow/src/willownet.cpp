#include <willow/device.hpp>
#include <willow/error.hpp>
#include <willow/ir.hpp>
#include <willow/popx/devicex.hpp>
#include <willow/willownet.hpp>

namespace willow {

WillowNet::WillowNet(std::string onnxModelFn,
                     const EarlyInfo &perk,
                     const DataFlow &df,
                     const std::vector<Loss *> &lossesIn,
                     const Optimizer *optimizerIn,
                     // Weights tensors which are not to be updated
                     const std::vector<TensorId> &cTens,
                     std::string logdir_,
                     const std::vector<std::string> &patternNames)

    : pir(new Ir({onnxModelFn,
                  perk,
                  df,
                  lossesIn,
                  optimizerIn,
                  cTens,
                  logdir_,
                  patternNames})) {}

void WillowNet::updateOptimizer(const Optimizer *optimizer) {
  pir->updateOptimizer(optimizer);
}

void WillowNet::setDevice(std::string deviceString) {
  // TODO : here, need macro around specific device types,
  // might not have enabled to build with them.
  if (deviceString == "IPU") {
    device_.reset(new popx::Devicex(pir.get()));
  } else {
    throw error("How to set device from " + deviceString + " ??? ");
  }
}

WillowNet::~WillowNet() = default;

void WillowNet::prepareDevice() { device_->prepare(); }

void WillowNet::weightsFromHost() { device_->weightsFromHost(); }

// write whatever optimizer tensors (learning rates,
// momentum, initial momentum tensors (zero)) there are to device
void WillowNet::optimizerFromHost() { device_->optimizerFromHost(); }

// For Poplar, this will involve reading and writing
// Poplar::Stream <--> these addresses.

void WillowNet::step(const StepIO &) {
  throw error("pop device not ready, step");
}

// write current model to ONNX file
void WillowNet::modelToHost(std::string) {
  throw error("pop device not ready, model to host");
}

} // namespace willow
