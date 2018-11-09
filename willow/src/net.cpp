#include <fstream>
#include <poponnx/device.hpp>
#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/net.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>

namespace willow {

Net::Net(std::string onnxModelFn,
         const EarlyInfo &perk,
         const DataFlow &df,
         const std::vector<Loss *> &lossesIn,
         const Optimizer *optimizerIn,
         // Weights tensors which are not to be updated
         const std::vector<TensorId> &cTens,
         std::string logdir_,
         const std::vector<std::string> &patternNames)

    : pir_(new Ir({onnxModelFn,
                   perk,
                   df,
                   lossesIn,
                   optimizerIn,
                   cTens,
                   logdir_,
                   patternNames})),
      device_(nullptr) {}

void Net::updateOptimizer(const Optimizer *optimizer) {
  pir_->updateOptimizer(optimizer);
}

void Net::setDevice(std::string deviceString) {
  // TODO(See task T5103) : here, need macro around specific device types,
  // might not have enabled to build with them. (in CMakeLists.txt
  // there is POPLAR_BACKEND option)

  if (deviceString == "IPU") {
    device_.reset(new popx::Devicex(pir_.get()));
  } else {
    throw error("How to set device from " + deviceString + " ??? ");
  }
}

// get the TensorInfo on a Tensor
TensorInfo Net::getInfo(TensorId id) const {
  TensorInfo info = pir_->tensors.get(id)->info;
  if (!info.isSet()) {
    throw error("TensorInfo for `" + id + "' not set");
  }
  return info;
}

Net::~Net() = default;

void Net::prepareDevice() { device_->prepare(); }

void Net::weightsFromHost() { device_->weightsFromHost(); }

// write whatever optimizer tensors (learning rates,
// momentum, initial momentum tensors (zero)) there are to device
void Net::optimizerFromHost() { device_->optimizerFromHost(); }

// For Poplar, this will involve reading and writing
// Poplar::Stream <--> these addresses and running a program
void Net::step(const StepIO &stepio) { device_->step(stepio); }

// write current model to ONNX file
void Net::modelToHost(std::string fn) {

  onnx::ModelProto model = pir_->getModel();

  std::map<TensorId, MutableVoidData> initMap;
  for (int init_index = 0; init_index < model.graph().initializer_size();
       ++init_index) {
    onnx::TensorProto &tp =
        *model.mutable_graph()->mutable_initializer(init_index);
    TensorId tenId = tp.name();
    initMap[tenId] = onnxutil::getMutableData(tp);
  }

  device_->weightsToHost(initMap);

  io::writeModel(model, fn);
}

} // namespace willow
