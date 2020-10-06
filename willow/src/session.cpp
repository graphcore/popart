// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <fstream>

#include <popart/builder_impl.hpp>
#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/onnxutil.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/exporter.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>
#include <popart/version.hpp>

#include <popart/poparttracepoint.hpp>

namespace popart {

Session::Session() {
  POPART_TRACEPOINT();
  logging::session::info("Popart version: {}", popart::core::versionString());
  logging::session::info("Popart release githash: {}",
                         popart::core::packageHash());
}

void Session::setDevice(std::shared_ptr<DeviceInfo> deviceInfo) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::setDevice({})", *deviceInfo);
  device_.reset(new popx::Devicex(ir, deviceInfo));
}

void Session::setRandomSeed(uint64_t seedValue) {
  POPART_TRACEPOINT();
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

uint64_t Session::getCycleCount(std::string id) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::getCycleCount()");
  if (!runCalled) {
    throw error("Must call run before getCycleCount.");
  }
  auto cycleCounts = device_->cycleCountTensorToHost();
  if (cycleCounts.find(id) != cycleCounts.end()) {
    // Always get cycle count from first replica
    return cycleCounts.at(id)[0];
  } else {
    throw error("Invalid id for cycle counter, '{}'. Make sure you have set "
                "SessionOption::hardwareInstrumentations correctly.",
                id);
  }
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

void Session::compileAndExport(std::string executablePath,
                               std::string weightsPath) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::compileAndExport()");

  if (executablePath.length() > 0) {
    executablePath = io::getCanonicalDirName(executablePath);
    io::assertDirectoryWritable(executablePath);
  }

  if (weightsPath.length() > 0) {
    weightsPath = io::getCanonicalDirName(weightsPath);
    io::assertDirectoryWritable(weightsPath);
  }

  device_->compileAndExport(executablePath, weightsPath);
}

void Session::prepareDevice() {
  POPART_TRACEPOINT();
  logging::session::trace("Session::prepareDevice()");

  device_->prepare();
}

void Session::weightsFromHost() {
  POPART_TRACEPOINT();
  logging::session::trace("Sessions::weightsFromHost");

  device_->weightsFromHost();
  weightsFromHostCalled = true;
}

void Session::weightsToHost() {
  POPART_TRACEPOINT();
  logging::session::trace("Session::weightsToHost");

  if (!device_) {
    throw error("Must call setDevice before {}", __func__);
  }

  device_->weightsToHost();
}

void Session::readWeights(const IWeightsIO &weightsIo) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::readWeights");

  if (!device_) {
    throw error("Must call setDevice before {}", __func__);
  }

  device_->readWeights(weightsIo);
}

void Session::writeWeights(const IWeightsIO &weightsIo) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::writeWeights");

  if (!device_) {
    throw error("Must call setDevice before {}", __func__);
  }

  device_->writeWeights(weightsIo);
}

void Session::exportInputs(IStepIO &stepIO,
                           int64_t num_elements,
                           const std::string &output_filename) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::exportInputs");
  exportStepIO(stepIO, *device_, num_elements, output_filename);
}

void Session::run(IStepIO &stepio, std::string debugName) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::run {}", debugName);
  if (!ir.canInfer()) {
    throw error("Trying to infer when not in inference mode");
  }

  if (ir.containsInitialisers() && ir.isTraining() &&
      weightsFromHostCalled == false) {
    throw error(
        "Must call weightsFromHost before run as the model has initializers "
        "and the session has been created in training mode");
  }

  device_->run(stepio, debugName);

  runCalled = true;
}

// write current model to ONNX file
void Session::modelToHost(const std::string &fn) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::modelToHost");

  ONNX_NAMESPACE::ModelProto model      = ir.getModel();
  ONNX_NAMESPACE::GraphProto *onnxgraph = model.mutable_graph();

  for (auto tId : ir.additionalModelProtoTensors) {
    // For additional tensors we want to save in the onnx modelproto, we copy
    // their info into across to the proto.
    if (ir.tensorExistsInInitialisers(tId)) {
      throw error("Tensor id {} already in initializers, duplicate tensor "
                  "Ids not allowed in onnx specification.",
                  tId);
    } else {
      ONNX_NAMESPACE::TensorProto *init = onnxgraph->add_initializer();
      init->set_name(tId);
      auto tensor = ir.getMainGraph().getTensors().get(tId);

      ConstVoidData cvData;
      cvData.data = tensor->tensorData()->data();
      cvData.info = tensor->info;
      BuilderImpl::populateTensorProtoFromConstVoidData(cvData, tId, init);
    }
  }

  std::map<TensorId, MutableVoidData> initMap;
  // For storing tensor data for externally stored tensors.
  std::map<TensorId, std::vector<char>> externalTensorBuffers;

  for (int init_index = 0; init_index < model.graph().initializer_size();
       ++init_index) {
    ONNX_NAMESPACE::TensorProto &tp =
        *model.mutable_graph()->mutable_initializer(init_index);
    TensorId tenId = tp.name();

    if (tp.has_data_location() &&
        tp.data_location() == ONNX_NAMESPACE::TensorProto::EXTERNAL) {
      // Initialise a new MutableVoidData object to write to from host weight
      // buffers
      MutableVoidData mvd;
      mvd.info = getInfo(tenId);
      std::vector<char> buffer(mvd.info.nbytes());
      externalTensorBuffers.emplace(tenId, buffer);
      mvd.data = reinterpret_cast<void *>(externalTensorBuffers[tenId].data());
      initMap[tenId] = mvd;
    } else {
      initMap[tenId] = onnxutil::getMutableData(tp);
    }
  }

  device_->weightsToHost(initMap);

  // Write data for externally saved weights to relevant locations on disk
  for (int init_index = 0; init_index < model.graph().initializer_size();
       ++init_index) {
    ONNX_NAMESPACE::TensorProto tp = model.graph().initializer(init_index);

    if (tp.has_data_location() &&
        tp.data_location() == ONNX_NAMESPACE::TensorProto::EXTERNAL) {
      TensorId tenId    = tp.name();
      auto externalInfo = onnxutil::ExternalTensorProtoInfo(tp);
      std::fstream ofs(externalInfo.location,
                       std::ofstream::binary | std::ios_base::out |
                           std::ios_base::in);
      if (!ofs.is_open()) {
        throw error("Trying to update initializer {}, stored in file {}, when "
                    "writing modelToHost. Failed to open file",
                    tenId,
                    fn);
      }

      if (externalInfo.offset > 0) {
        ofs.seekp(externalInfo.offset, std::ios::beg);
      }
      ofs.write(static_cast<const char *>(initMap[tenId].data),
                externalInfo.length);
      ofs.close();
    }
  }

  io::writeModel(model, fn);

  if (!ir.getSessionOptions().constantWeights ||
      ir.getExecutionMode() != Ir::ExecutionMode::Inference) {
    // Weights in ir, device, and disk now all match
    ir.resetWeights(model);
  }
}

std::string Session::getSummaryReport(bool resetProfile) const {
  POPART_TRACEPOINT();
  logging::session::trace("Session::getSummaryReport");

  return device_->getSummaryReport(resetProfile);
}

std::string Session::getGraphReport(bool useCbor) const {
  POPART_TRACEPOINT();
  logging::session::trace("Session::getGraphReport");
  return device_->getGraphReport(useCbor);
}

std::string Session::getExecutionReport(bool useCbor, bool resetProfile) const {
  POPART_TRACEPOINT();
  logging::session::trace("Session::getExecutionReport");
  return device_->getExecutionReport(useCbor, resetProfile);
}

std::string Session::getSerializedGraph() const {
  POPART_TRACEPOINT();
  logging::session::trace("Session::getSerializedGraph");
  return device_->getSerializedGraph();
}

TensorTileMap Session::getTensorTileMap() const {
  logging::session::trace("Session::getTensorTileMap");

  return device_->getTensorTileMap();
}

void Session::resetHostWeights(
    const std::string &modelProtoOrFilename,
    const bool ignoreWeightsInModelWithoutCorrespondingHostWeight) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::resetHostWeights");
  if (ir.getSessionOptions().constantWeights &&
      ir.getExecutionMode() == Ir::ExecutionMode::Inference) {
    throw error("Cannot call resetHostWeights when constantWeights is set");
  }
  auto modelProto = onnxutil::getModelProto(modelProtoOrFilename);
  ir.resetWeights(modelProto,
                  ignoreWeightsInModelWithoutCorrespondingHostWeight);

  // After the weights has been reset they must be rewritten to the target
  weightsFromHostCalled = false;
}

std::string Session::serializeIr(IrSerializationFormat format) {
  (void)format;
  std::stringstream ss;
  ir.serialise(Ir::SerialiseFormat::JSON, ss);
  return ss.str();
}

InferenceSession::InferenceSession() : Session() { POPART_TRACEPOINT(); }

InferenceSession::~InferenceSession() = default;

void InferenceSession::configureFromOnnx(
    const std::string &modelProtoOrFilename,
    const DataFlow &df,
    const InputShapeInfo &perk,
    std::shared_ptr<DeviceInfo> deviceInfo,
    const SessionOptions &userOptions,
    const Patterns &patterns) {
  POPART_TRACEPOINT();

  logging::session::trace("InferenceSession::configureFromOnnx");

  auto modelProto = onnxutil::getModelProto(modelProtoOrFilename);

  ir.prepare(
      {modelProto, perk, df, {}, nullptr, *deviceInfo, userOptions, patterns});
}

std::unique_ptr<InferenceSession>
InferenceSession::createFromOnnxModel(const std::string &model,
                                      const DataFlow &dataFlow,
                                      std::shared_ptr<DeviceInfo> deviceInfo,
                                      const InputShapeInfo &inputShapeInfo,
                                      const SessionOptions &userOptions,
                                      const Patterns &patterns) {
  POPART_TRACEPOINT();
  logging::session::trace("InferenceSession::createFromOnnx");

  if (!deviceInfo) {
    throw error("Must pass a valid deviceInfo to "
                "InferenceSession::createFromOnnxModel");
  }

  auto session = std::unique_ptr<InferenceSession>(new InferenceSession());
  session->configureFromOnnx(
      model, dataFlow, inputShapeInfo, deviceInfo, userOptions, patterns);

  session->setDevice(deviceInfo);

  return session;
}

TrainingSession::TrainingSession() : Session() { POPART_TRACEPOINT(); }

TrainingSession::~TrainingSession() = default;

void TrainingSession::configureFromOnnx(const std::string &modelProtoOrFilename,
                                        const DataFlow &df,
                                        const TensorId &lossIn,
                                        const Optimizer &optimizerIn,
                                        const InputShapeInfo &perk,
                                        std::shared_ptr<DeviceInfo> deviceInfo,
                                        const SessionOptions &userOptions,
                                        const Patterns &patterns) {
  POPART_TRACEPOINT();
  logging::session::trace("TrainingSession::configureFromOnnx");

  auto modelProto = onnxutil::getModelProto(modelProtoOrFilename);

  ir.prepare({modelProto,
              perk,
              df,
              lossIn,
              &optimizerIn,
              *deviceInfo,
              userOptions,
              patterns});
}

std::unique_ptr<TrainingSession>
TrainingSession::createFromOnnxModel(const std::string &model,
                                     const DataFlow &dataFlow,
                                     const TensorId &loss,
                                     const Optimizer &optimizer,
                                     std::shared_ptr<DeviceInfo> deviceInfo,
                                     const InputShapeInfo &inputShapeInfo,
                                     const SessionOptions &userOptions,
                                     const Patterns &patterns) {
  POPART_TRACEPOINT();
  logging::session::trace("TrainingSession::createFromOnnx");

  if (!deviceInfo) {
    throw error(
        "Must pass a valid deviceInfo to TrainingSession::createFromOnnxModel");
  }

  auto session = std::unique_ptr<TrainingSession>(new TrainingSession());
  session->configureFromOnnx(model,
                             dataFlow,
                             loss,
                             optimizer,
                             inputShapeInfo,
                             deviceInfo,
                             userOptions,
                             patterns);

  session->setDevice(deviceInfo);

  return session;
}

void TrainingSession::updateOptimizerFromHost(const Optimizer *optimizer) {
  POPART_TRACEPOINT();
  logging::session::trace("TrainingSession::updateOptimizerFromHost");
  ir.updateOptimizer(*optimizer);

  // There has been a change to the TensorData of the optimizer tensors
  // on the host, but there wont be an equivalent update to the device-side
  // tensors until optimizerFromHost() is called.

  // write whatever optimizer tensors (learning rates,
  // momentum, initial momentum tensors) there are to device
  device_->optimizerFromHost();
}

const std::vector<std::string> &
TrainingSession::getHostReduceStreamIds() const {
  return device_->getHostReduceStreamIds();
}

const std::map<std::string, poplar::RemoteBuffer> &
TrainingSession::getHostReduceRemoteBuffers() const {
  return device_->getHostReduceRemoteBuffers();
}

void TrainingSession::connectStreamToCallback(
    const std::string &streamHandle,
    std::function<void(void *)> callback,
    unsigned index) {
  POPART_TRACEPOINT();
  device_->connectStreamToCallback(streamHandle, callback, index);
}

void TrainingSession::copyFromRemoteBuffer(const std::string &buffer,
                                           void *w,
                                           int repeat_index,
                                           unsigned replication_index) {
  POPART_TRACEPOINT();
  device_->copyFromRemoteBuffer(buffer, w, repeat_index, replication_index);
}

void TrainingSession::copyToRemoteBuffer(void *w,
                                         const std::string &buffer,
                                         int repeat_index,
                                         unsigned replication_index) {
  POPART_TRACEPOINT();
  device_->copyToRemoteBuffer(w, buffer, repeat_index, replication_index);
}

} // namespace popart
