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

Net::Net(const std::string &modelProtoOrFilename,
         const EarlyInfo &perk,
         const DataFlow &df,
         const std::vector<Loss *> &lossesIn,
         const Optimizer *optimizerIn,
         const std::vector<TensorId> &cTens,
         std::string logdir,
         const std::vector<std::string> &patternNames)

    : device_(nullptr) {

  onnx::ModelProto modelProto;
  try {
    modelProto = io::getModelFromFile(modelProtoOrFilename);
  } catch (const error &) {
    modelProto = io::getModelFromString(modelProtoOrFilename);
    ;
  }

  pir_.reset(new Ir({modelProto,
                     perk,
                     df,
                     lossesIn,
                     optimizerIn,
                     cTens,
                     logdir,
                     patternNames}));
}

void Net::updateOptimizer(const Optimizer *optimizer) {
  pir_->updateOptimizer(optimizer);
}

void Net::setDevice(const std::string &deviceString) {
  if (deviceString == "IPU") {
    device_.reset(new popx::Devicex(pir_.get()));
  } else {
    throw error("Unrecognised device type: " + deviceString);
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
void Net::modelToHost(const std::string &fn) {

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
