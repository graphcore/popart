#include <poponnx/device.hpp>
#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/util.hpp>

namespace willow {

Session::Session() {}

void Session::configureFromOnnx(const std::string &modelProtoOrFilename,
                                const EarlyInfo &perk,
                                const DataFlow &df,
                                const std::vector<Loss *> &lossesIn,
                                const Optimizer *optimizerIn,
                                const std::vector<TensorId> &cTens,
                                std::string logdir,
                                const SessionOptions &userOptions,
                                const std::vector<std::string> &patternNames) {

  onnx::ModelProto modelProto;
  try {
    modelProto = io::getModelFromFile(modelProtoOrFilename);
  } catch (const error &) {
    modelProto = io::getModelFromString(modelProtoOrFilename);
  }

  ir.prepare({modelProto,
              perk,
              df,
              lossesIn,
              optimizerIn,
              cTens,
              logdir,
              userOptions,
              patternNames});
}

std::unique_ptr<Session>
Session::createFromOnnxModel(const std::string &model,
                             const EarlyInfo &earlyInfo,
                             const DataFlow &dataFlow,
                             const std::vector<Loss *> &losses,
                             const Optimizer *optimizer,
                             const std::vector<std::string> &cTens,
                             std::string logdir,
                             const SessionOptions &userOptions,
                             const std::vector<std::string> &patternNames) {

  // Note : Can not use make_unique as the implementation can not acces the
  // private constructor
  auto session = std::unique_ptr<Session>(new Session());
  session->configureFromOnnx(model,
                             earlyInfo,
                             dataFlow,
                             losses,
                             optimizer,
                             cTens,
                             logdir,
                             userOptions,
                             patternNames);
  return session;
}

void Session::updateOptimizer(const Optimizer *optimizer) {
  ir.updateOptimizer(optimizer);
}

void Session::setDevice(const std::string &deviceString) {
  if (deviceString == "IPU") {
    device_.reset(new popx::Devicex(ir));
  } else {
    throw error("Unrecognised device type: " + deviceString);
  }
}

// get the TensorInfo on a Tensor
TensorInfo Session::getInfo(TensorId id) const {
  TensorInfo info = ir.getTensors().get(id)->info;
  if (!info.isSet()) {
    throw error("TensorInfo for `" + id + "' not set");
  }
  return info;
}

Session::~Session() = default;

void Session::prepareDevice() { device_->prepare(); }

void Session::weightsFromHost() { device_->weightsFromHost(); }

// write whatever optimizer tensors (learning rates,
// momentum, initial momentum tensors (zero)) there are to device
void Session::optimizerFromHost() { device_->optimizerFromHost(); }

void Session::train(const StepIO &stepio) { device_->step(stepio); }

void Session::evaluate(const StepIO &) { return; }

void Session::infer(const StepIO &) { return; }

// write current model to ONNX file
void Session::modelToHost(const std::string &fn) {

  onnx::ModelProto model = ir.getModel();

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
