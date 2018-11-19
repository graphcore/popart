#include <poponnx/device.hpp>
#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
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

  logging::session::trace("Session::configureFromOnnx");

  onnx::ModelProto modelProto;
  if (io::isRegularFile(modelProtoOrFilename)) {
    modelProto = io::getModelFromFile(modelProtoOrFilename);
  } else {
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

  // Needs to be the first call to initialise the logging settings
  logging::configure(userOptions.loggingOptions);

  logging::session::trace("Session::createFromOnnx");

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
  logging::session::trace("Session::updateOptimzier");
  ir.updateOptimizer(optimizer);
}

void Session::setDevice(const std::string &deviceString) {
  logging::session::trace("Session::setDevice({})", deviceString);
  if (deviceString == "IPU") {
    device_.reset(new popx::Devicex(ir));
  } else {
    throw error("Unrecognised device type: " + deviceString);
  }
}

// get the TensorInfo on a Tensor
TensorInfo Session::getInfo(TensorId id) const {
  logging::session::trace("Session::getInfo({})", id);
  TensorInfo info = ir.getTensors().get(id)->info;
  if (!info.isSet()) {
    throw error("TensorInfo for `" + id + "' not set");
  }
  return info;
}

Session::~Session() = default;

void Session::prepareDevice() {
  logging::session::trace("Session::prepareDevice()");
  device_->prepare();
}

void Session::weightsFromHost() {
  logging::session::trace("Sessions::weightsFromHost");
  device_->weightsFromHost();
}

// write whatever optimizer tensors (learning rates,
// momentum, initial momentum tensors (zero)) there are to device
void Session::optimizerFromHost() {
  logging::session::trace("Session::optimzierFromHost");
  device_->optimizerFromHost();
}

void Session::train(const StepIO &stepio) {
  logging::session::trace("Session::train");
  device_->train(stepio);
}

void Session::evaluate(const StepIO &stepio) {
  logging::session::trace("Session::evaluate");
  device_->evaluate(stepio);
}

void Session::infer(const StepIO &stepio) {
  logging::session::trace("Session::infer");
  device_->infer(stepio);
}

// write current model to ONNX file
void Session::modelToHost(const std::string &fn) {
  logging::session::trace("Session::modelToHost");

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

std::string Session::getSummaryReport() const {
  logging::session::trace("Session::getSummaryReport");
  return device_->getSummaryReport();
}

std::string Session::getGraphReport() const {
  logging::session::trace("Session::getGraphReport");
  return device_->getGraphReport();
}

std::string Session::getExecutionReport() const {
  logging::session::trace("Session::getExecutionReport");
  return device_->getExecutionReport();
}

} // namespace willow
