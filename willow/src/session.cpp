#include <poponnx/error.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/logging.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/util.hpp>
#include <poponnx/version.hpp>

namespace poponnx {

Session::Session() {
  logging::session::info("Poponnx version: {}", poponnx::core::versionString());
  logging::session::info("Poponnx release githash: {}",
                         poponnx::core::packageHash());
}

void Session::setDevice(std::shared_ptr<DeviceInfo> deviceInfo) {
  logging::session::trace("Session::setDevice({})", *deviceInfo);
  device_.reset(new popx::Devicex(ir, deviceInfo));
}

// get the TensorInfo on a Tensor
TensorInfo Session::getInfo(TensorId id) const {
  logging::session::trace("Session::getInfo({})", id);
  TensorInfo info = ir.getMainGraph().getTensors().get(id)->info;
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
  weightsFromHostCalled = true;
}

void Session::weightsToHost() {
  logging::session::trace("Session::weightsToHost");

  if (!device_) {
    throw error("Must call setDevice before {}", __func__);
  }

  device_->weightsToHost();
}

void Session::readWeights(const IWeightsIO &weightsIo) {
  logging::session::trace("Session::readWeights");

  if (!device_) {
    throw error("Must call setDevice before {}", __func__);
  }

  device_->readWeights(weightsIo);
}

void Session::writeWeights(const IWeightsIO &weightsIo) {
  logging::session::trace("Session::writeWeights");

  if (!device_) {
    throw error("Must call setDevice before {}", __func__);
  }

  device_->writeWeights(weightsIo);
}

void Session::run(const IStepIO &stepio) {
  logging::session::trace("Session::run");
  if (!ir.canInfer()) {
    throw error("Trying to infer when not in inference mode");
  }

  if (ir.containsInitialisers() && ir.isTraining() &&
      weightsFromHostCalled == false) {
    throw error(
        "Must call weightsFromHost before run as the model has initializers "
        "and the session has been created in training mode");
  }

  device_->run(stepio);
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

  if (!ir.getSessionOptions().constantWeights ||
      ir.getExecutionMode() != Ir::ExecutionMode::INFERENCE) {
    // Weights in ir, device, and disk now all match
    ir.resetWeights(model);
  }
}

std::string Session::getSummaryReport() const {
  logging::session::trace("Session::getSummaryReport");

  return device_->getSummaryReport();
}

std::string Session::getGraphReport(bool use_cbor) const {
  logging::session::trace("Session::getGraphReport");
  return device_->getGraphReport(use_cbor);
}

std::string Session::getExecutionReport(bool use_cbor) const {
  logging::session::trace("Session::getExecutionReport");
  return device_->getExecutionReport(use_cbor);
}

std::string Session::getSerializedGraph() const {
  logging::session::trace("Session::getSerializedGraph");
  return device_->getSerializedGraph();
}

TensorTileMap Session::getTensorTileMap() const {
  logging::session::trace("Session::getTensorTileMap");

  return device_->getTensorTileMap();
}

void Session::resetHostWeights(const std::string &modelProtoOrFilename) {
  logging::session::trace("Session::resetHostWeights");
  if (ir.getSessionOptions().constantWeights &&
      ir.getExecutionMode() == Ir::ExecutionMode::INFERENCE) {
    throw error("Cannot call resetHostWeights when constantWeights is set");
  }
  auto modelProto = onnxutil::getModelProto(modelProtoOrFilename);
  ir.resetWeights(modelProto);

  // After the weights has been reset they must be rewritten to the target
  weightsFromHostCalled = false;
}

InferenceSession::InferenceSession() : Session() {}

InferenceSession::~InferenceSession() = default;

void InferenceSession::configureFromOnnx(
    const std::string &modelProtoOrFilename,
    const DataFlow &df,
    const std::vector<Loss *> &losses,
    const InputShapeInfo &perk,
    std::shared_ptr<DeviceInfo> deviceInfo,
    const SessionOptions &userOptions,
    const Patterns &patterns) {

  logging::session::trace("InferenceSession::configureFromOnnx");

  auto modelProto = onnxutil::getModelProto(modelProtoOrFilename);

  ir.prepare({modelProto,
              perk,
              df,
              losses,
              nullptr,
              *deviceInfo,
              userOptions,
              patterns});
}

std::unique_ptr<InferenceSession>
InferenceSession::createFromOnnxModel(const std::string &model,
                                      const DataFlow &dataFlow,
                                      std::shared_ptr<DeviceInfo> deviceInfo,
                                      const std::vector<Loss *> &losses,
                                      const InputShapeInfo &inputShapeInfo,
                                      const SessionOptions &userOptions,
                                      const Patterns &patterns) {

  logging::session::trace("InferenceSession::createFromOnnx");

  if (!deviceInfo) {
    throw error("Must pass a valid deviceInfo to "
                "InferenceSession::createFromOnnxModel");
  }

  auto session = std::unique_ptr<InferenceSession>(new InferenceSession());
  session->configureFromOnnx(model,
                             dataFlow,
                             losses,
                             inputShapeInfo,
                             deviceInfo,
                             userOptions,
                             patterns);

  session->setDevice(deviceInfo);

  return session;
}

TrainingSession::TrainingSession() : Session() {}

TrainingSession::~TrainingSession() = default;

void TrainingSession::configureFromOnnx(const std::string &modelProtoOrFilename,
                                        const DataFlow &df,
                                        const std::vector<Loss *> &lossesIn,
                                        const Optimizer &optimizerIn,
                                        const InputShapeInfo &perk,
                                        std::shared_ptr<DeviceInfo> deviceInfo,
                                        const SessionOptions &userOptions,
                                        const Patterns &patterns) {

  logging::session::trace("TrainingSession::configureFromOnnx");

  auto modelProto = onnxutil::getModelProto(modelProtoOrFilename);

  ir.prepare({modelProto,
              perk,
              df,
              lossesIn,
              &optimizerIn,
              *deviceInfo,
              userOptions,
              patterns});
}

std::unique_ptr<TrainingSession>
TrainingSession::createFromOnnxModel(const std::string &model,
                                     const DataFlow &dataFlow,
                                     const std::vector<Loss *> &losses,
                                     const Optimizer &optimizer,
                                     std::shared_ptr<DeviceInfo> deviceInfo,
                                     const InputShapeInfo &inputShapeInfo,
                                     const SessionOptions &userOptions,
                                     const Patterns &patterns) {

  logging::session::trace("TrainingSession::createFromOnnx");

  if (!deviceInfo) {
    throw error(
        "Must pass a valid deviceInfo to TrainingSession::createFromOnnxModel");
  }

  auto session = std::unique_ptr<TrainingSession>(new TrainingSession());
  session->configureFromOnnx(model,
                             dataFlow,
                             losses,
                             optimizer,
                             inputShapeInfo,
                             deviceInfo,
                             userOptions,
                             patterns);

  session->setDevice(deviceInfo);

  return session;
}

void TrainingSession::updateOptimizer(const Optimizer *optimizer) {
  logging::session::trace("TrainingSession::updateOptimizer");
  ir.updateOptimizer(optimizer);
}

// write whatever optimizer tensors (learning rates,
// momentum, initial momentum tensors (zero)) there are to device
void TrainingSession::optimizerFromHost() {
  logging::session::trace("TrainingSession::optimizerFromHost");

  device_->optimizerFromHost();
}
} // namespace poponnx
