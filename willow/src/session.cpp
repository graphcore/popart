#include <popart/builder_impl.hpp>
#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/onnxutil.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>
#include <popart/version.hpp>

namespace popart {

Session::Session() {
  logging::session::info("Popart version: {}", popart::core::versionString());
  logging::session::info("Popart release githash: {}",
                         popart::core::packageHash());
}

void Session::setDevice(std::shared_ptr<DeviceInfo> deviceInfo) {
  logging::session::trace("Session::setDevice({})", *deviceInfo);
  device_.reset(new popx::Devicex(ir, deviceInfo));
}

void Session::setRandomSeed(uint64_t seedValue) {
  logging::session::trace("Session::setRandomSeed({})", seedValue);
  if (!ir.requiresRandomSeed()) {
    logging::session::warn("Trying to set the random seed, but this session "
                           "has no random behaviour. Doing nothing.");
    return;
  }
  // Set seed value on host
  ir.setRandomSeedValue(seedValue);

  // ... Then stream to device
  if (!device_->prepareHasBeenCalled()) {
    throw error("Devicex::prepare() must be called before "
                "Devicex::setRandomSeedFromHost(uint64_t) is called.");
  }
  device_->setRandomSeedFromHost();
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

void Session::run(IStepIO &stepio) {
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

  if (!ir.optimizerTensors().empty() && ir.isTraining() &&
      optimizerFromHostCalledSinceLastUpdate == false) {
    throw error(
        "Must call optimizerFromHost before run as the optimizer tensor values "
        "have not been written to the device");
  }

  device_->run(stepio);
}

// write current model to ONNX file
void Session::modelToHost(const std::string &fn) {
  logging::session::trace("Session::modelToHost");

  onnx::ModelProto model      = ir.getModel();
  onnx::GraphProto *onnxgraph = model.mutable_graph();

  for (auto tId : ir.additionalModelProtoTensors) {
    // For additional tensors we want to save in the onnx modelproto, we copy
    // their info into across to the proto.
    if (ir.tensorExistsInInitialisers(tId)) {
      throw error("Tensor id {} already in initializers, duplicate tensor "
                  "Ids not allowed in onnx specification.",
                  tId);
    } else {
      onnx::TensorProto *init = onnxgraph->add_initializer();
      init->set_name(tId);
      auto tensor = ir.getMainGraph().getTensors().get(tId);

      ConstVoidData cvData;
      cvData.data = tensor->tensorData()->data();
      cvData.info = tensor->info;
      BuilderImpl::populateTensorProtoFromConstVoidData(cvData, tId, init);
    }
  }

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

std::string Session::serializeIr(IrSerializationFormat format) {
  (void)format;
  std::stringstream ss;
  ir.serialise(Ir::SerialiseFormat::JSON, ss);
  return ss.str();
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
  ir.updateOptimizer(*optimizer);

  // There has been a change to the TensorData of the optimizer tensors
  // on the host, but there wont be an equivalent update to the device-side
  // tensors until optimizerFromHost() is called.
  optimizerFromHostCalledSinceLastUpdate = false;
}

// write whatever optimizer tensors (learning rates,
// momentum, initial momentum tensors) there are to device
void TrainingSession::optimizerFromHost() {
  logging::session::trace("TrainingSession::optimizerFromHost");

  device_->optimizerFromHost();
  optimizerFromHostCalledSinceLastUpdate = true;
}

const Ir &TrainingSession::getIr() const { return ir; }

const std::vector<std::pair<std::string, std::string>> &
TrainingSession::getGradAndVarStreamIds() const {
  return device_->getGradAndVarStreamIds();
}

void TrainingSession::connectStreamToCallback(
    const std::string &streamHandle,
    std::function<void(void *)> callback) {
  device_->connectStreamToCallback(streamHandle, callback);
}

} // namespace popart
