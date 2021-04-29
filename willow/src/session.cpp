// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <fstream>

#include <boost/filesystem.hpp>

#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/onnxutil.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/executablex.hpp>
#include <popart/popx/executablexserialization.hpp>
#include <popart/session.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/transforms/randomsetup.hpp>
#include <popart/util.hpp>
#include <popart/version.hpp>

#include <popart/poparttracepoint.hpp>

namespace popart {

namespace {
HashesMap getCacheEntries(const std::string &cachePath) {
  HashesMap cacheEntries;
  if (!boost::filesystem::is_directory(cachePath)) {
    return cacheEntries;
  }
  for (auto &entry : boost::filesystem::directory_iterator(cachePath)) {
    if (boost::filesystem::is_regular_file(entry)) {
      std::ifstream in(entry.path().string(), std::ifstream::binary);
      try {
        auto hash = popart::popx::serialization::readExecutableHash(in);
        cacheEntries.emplace(hash, entry.path().string());
      } catch (const std::exception &e) {
        logging::session::trace("Ignoring invalid cache file {}: {}",
                                entry.path().string(),
                                e.what());
      }
    }
  }
  return cacheEntries;
}
} // namespace

void Session::ctorCommonLogic() {
  POPART_TRACEPOINT();
  logging::session::info("Popart version: {}", popart::core::versionString());
  logging::session::info("Popart release githash: {}",
                         popart::core::packageHash());
}

Session::Session() { ctorCommonLogic(); }

Session::Session(Ir ir_, std::shared_ptr<DeviceInfo> deviceInfo)
    : ir(std::move(ir_)) {
  ctorCommonLogic();
  setDevice(std::move(deviceInfo));
  initProgressLogger(ir.getSessionOptions());
}

void Session::setDevice(std::shared_ptr<DeviceInfo> deviceInfo) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::setDevice({})", *deviceInfo);
  deviceInfo_ = deviceInfo;

  if (ir.hashMatched()) {
    // The executable will be loaded during prepareDevice
    return;
  }

  lowering_.reset(new popx::IrLowering(ir, deviceInfo));
  executable_ = popx::Executablex::createFromLoweredIr(*lowering_);
  device_.reset(new popx::Devicex(*executable_, deviceInfo));
}

std::vector<uint32_t> Session::getRNGState() {
  if (!ir.getSessionOptions().enableLoadAndOffloadRNGState) {
    throw error("Trying to get the RNG state, but the session option "
                "enableLoadAndOffloadRNGState must be set to True.");
  }
  if (!device_->prepareHasBeenCalled()) {
    throw error("Devicex::prepare() must be called before "
                "Devicex::getRngStateToHost is called.");
  }
  std::vector<uint32_t> seedValue;
  seedValue = device_->getRngStateToHost();
  return seedValue;
}

bool Session::tryLoadExecutable() {
  logging::session::trace("Session::tryLoadExecutable()");
  if (false == ir.isPrepared()) {
    throw error("Ir::prepare() must be called before trying to load a cached "
                "executable");
  }
  const SessionOptions &userOptions = ir.getSessionOptions();

  if (false == Ir::usingEngineCache(userOptions, deviceInfo_.get())) {
    return false;
  }

  if (false == ir.hashMatched()) {
    return false;
  }

  auto popartCachePath = cacheEntries.at(ir.getHash());
  std::ifstream executableFs(popartCachePath, std::ifstream::binary);
  if (executableFs.is_open()) {
    logging::session::info("Loading serialized PopART executable from {}",
                           popartCachePath);
    try {
      loadExecutableFromStream(executableFs);
      return true;
    } catch (const std::exception &e) {
      logging::session::warn(
          "Failed to load cached PopART executable from {}: {}",
          popartCachePath,
          e.what());
      return false;
    }
  }
  logging::session::info("Could not open file {}", popartCachePath);
  return false;
}

void Session::loadExecutableFromFile(std::string filename) {
  std::ifstream executableFs(filename, std::ifstream::binary);
  if (!executableFs.is_open()) {
    throw error("Could not open file {}", filename);
  }
  logging::session::info("Loading serialized PopART executable from {}",
                         filename);
  try {
    loadExecutableFromStream(executableFs);
  } catch (...) {
    logging::session::err(
        "Failed to load serialized PopART executable from {}:", filename);
    throw;
  }
}

void Session::loadExecutableFromStream(std::istream &in) {
  bool skipGraphCompilation = true;
  lowering_.reset(new popx::IrLowering(ir, deviceInfo_, skipGraphCompilation));

  lowering_->loadPoplarExecutable(in);

  executable_ = popx::serialization::deserializeExecutable(in, ir, *lowering_);

  device_.reset(new popx::Devicex(*executable_, deviceInfo_));

  popx::serialization::moveStreamToEnd(in);
}

void Session::assertExecutableLoaded() const {
  if (executable_ == nullptr) {
    throw error("There is no executatable. Try calling Session::prepareDevice "
                "first.");
  }
}

void Session::setRNGState(const std::vector<uint32_t> stateValue) {
  if (!ir.getSessionOptions().enableLoadAndOffloadRNGState) {
    throw error("Trying to set the RNG state, but the session option "
                "enableLoadAndOffloadRNGState must be set to True.");
  }
  // Set seed value on host
  device_->setRngStateValue(stateValue);

  // ... Then stream to device
  if (!device_->prepareHasBeenCalled()) {
    throw error("Devicex::prepare() must be called before "
                "Devicex::setRngStateFromHost is called.");
  }
  device_->setRngStateFromHost();
}

void Session::setRandomSeed(uint64_t seedValue) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::setRandomSeed({})", seedValue);
  if (!ir.getRequiresRandomSeed()) {
    logging::session::warn("Trying to set the random seed, but this session "
                           "has no random behaviour. Doing nothing.");
    return;
  }

  assertExecutableLoaded();

  // Set seed value on host
  executable_->setRandomSeedValue(seedValue);

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
  assertExecutableLoaded();
  TensorInfo info = executable_->getTensor(id)->info;
  if (!info.isSet()) {
    throw error("TensorInfo for `" + id + "' not set");
  }
  return info;
}

bool Session::hasInfo(TensorId id) const {
  assertExecutableLoaded();

  if (!executable_->containsTensor(id)) {
    return false;
  }

  TensorInfo info = executable_->getTensor(id)->info;
  return info.isSet();
}

Session::~Session() = default;

void Session::compileAndExport(std::ostream &out) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::compileAndExport()");

  // TODO(T34668): Ideally we'd just call prepareDevice() here
  // but we can't keep a copy of the poplar Executable or retrieve
  // it later so while we're waiting for this to be implemented
  // in Poplar we need to build the executable here.
  if (!tryLoadExecutable()) {
    lowering_->prepareGraph();
  }

  if (!device_->getDeviceInfo()->canCompileOffline()) {
    std::ostringstream oss;
    oss << device_->getDeviceInfo()->getType();

    throw error("Offline compilation is not supported for device type {}",
                oss.str());
  }

  auto poplarExecutable = executable_->getPoplarExecutable();
  executable_->serialize(poplarExecutable, out);
}

void Session::compileAndExport(std::string filename) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::compileAndExport()");

  // TODO(T34668): Ideally we'd just call prepareDevice() here
  // but we can't keep a copy of the poplar Executable or retrieve
  // it later so while we're waiting for this to be implemented
  // in Poplar we need to build the executable here.
  if (!tryLoadExecutable()) {
    lowering_->prepareGraph();
  }

  if (!device_->getDeviceInfo()->canCompileOffline()) {
    std::ostringstream oss;
    oss << device_->getDeviceInfo()->getType();

    throw error("Offline compilation is not supported for device type {}",
                oss.str());
  }

  auto poplarExecutable = executable_->getPoplarExecutable();
  executable_->serialize(poplarExecutable, filename);
}

void Session::prepareDevice(bool loadEngine) {
  POPART_TRACEPOINT();
  if (!tryLoadExecutable()) {
    lowering_->prepareGraph();
  }

  logging::session::trace("Session::prepareDevice()");
  if (!device_) {
    throw error("Must call setDevice before {}", __func__);
  }
  device_->prepare();

  if (ir.getSessionOptions().compileEngine && loadEngine) {
    loadEngineAndConnectStreams();
  }
}

void Session::loadEngineAndConnectStreams() {
  POPART_TRACEPOINT();

  logging::session::trace("Session::loadEngineAndConnectStreams()");
  if (!device_) {
    throw error("Must call setDevice before {}", __func__);
  }
  device_->loadEngineAndConnectStreams();
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

void Session::connectStreamToCallback(const std::string &streamHandle,
                                      std::function<void(void *)> callback,
                                      unsigned index) {
  POPART_TRACEPOINT();
  device_->connectStreamToCallback(streamHandle, callback, index);
}

void Session::connectStream(const std::string &streamHandle, void *buffer) {
  POPART_TRACEPOINT();
  device_->connectStream(streamHandle, buffer);
}

void Session::run(IStepIO &stepio, std::string debugName) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::run {}", debugName);
  if (device_->getDeviceInfo()->getConnectionType() ==
      DeviceConnectionType::Never) {
    throw error("Offline IPU device is not configured for execution");
  }
  if (!ir.canInfer()) {
    throw error("Trying to infer when not in inference mode");
  }

  if (weightsFromHostCalled == false && ir.containsInitialisers() &&
      ir.isTraining()) {
    throw error(
        "Must call weightsFromHost before run as the model has initializers "
        "and the session has been created in training mode");
  }
  device_->run(stepio, debugName);

  runCalled = true;
}

void Session::updateExternallySavedTensorLocations(
    const std::string &fromLocation,
    const std::string &toLocation) {
  // Check that toLocation does not exist
  if (boost::filesystem::exists(toLocation)) {
    throw error("Updating externally saved tensor location from file '{}' to "
                "file '{}', but file '{}' already exists",
                fromLocation,
                toLocation,
                toLocation);
  }

  // Check that fromLocation exists
  if (!boost::filesystem::exists(fromLocation)) {
    throw error("Updating externally saved tensor location from file '{}' to "
                "file '{}', but file '{}' does not exist",
                fromLocation,
                toLocation,
                fromLocation);
  }

  ONNX_NAMESPACE::ModelProto model = ir.getModel();
  std::vector<TensorId> tIds;
  for (int init_index = 0; init_index < model.graph().initializer_size();
       ++init_index) {
    ONNX_NAMESPACE::TensorProto tp = model.graph().initializer(init_index);

    if (tp.has_data_location() &&
        tp.data_location() == ONNX_NAMESPACE::TensorProto::EXTERNAL) {
      if (onnxutil::ExternalTensorProtoInfo(tp).location == fromLocation) {
        logging::session::debug("Changing the external data location for "
                                "tensor '{}' from '{}' to '{}'",
                                tp.name(),
                                fromLocation,
                                toLocation);
        tIds.push_back(tp.name());
      }
    }
  }

  if (tIds.empty()) {
    throw error("No ONNX model initializers have external location set to '{}'",
                fromLocation);
  }

  // Save the external data of tensors from the Ir's ONNX model to their
  // new locations
  onnxutil::saveInitializersExternally(
      model,
      tIds,
      toLocation,
      false, // Writing to a new file
      true); // Updating an existing externally saved tensor

  // Update the external tensor info of the Ir's ONNX model, so that
  // modelToHost will write tensor data to the new location, fn.
  for (auto tId : tIds) {
    ir.setExternalTensorDataInfo(tId, onnxutil::getTensorProto(model, tId));
  }
}

// write current model to ONNX file
void Session::modelToHost(const std::string &fn) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::modelToHost");

  assertExecutableLoaded();

  ONNX_NAMESPACE::ModelProto model = ir.getModel();

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
    // Weights in executable, device, and disk now all match
    executable_->resetWeights(model);
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

void Session::resetHostWeights(
    const std::string &modelProtoOrFilename,
    const bool ignoreWeightsInModelWithoutCorrespondingHostWeight) {
  POPART_TRACEPOINT();

  assertExecutableLoaded();

  logging::session::trace("Session::resetHostWeights");
  if (ir.getSessionOptions().constantWeights &&
      ir.getExecutionMode() == Ir::ExecutionMode::Inference) {
    throw error("Cannot call resetHostWeights when constantWeights is set");
  }
  auto modelProto = onnxutil::getModelProto(modelProtoOrFilename);
  executable_->resetWeights(modelProto,
                            ignoreWeightsInModelWithoutCorrespondingHostWeight);

  // After the weights has been reset they must be rewritten to the target
  weightsFromHostCalled = false;
}

std::string Session::serializeIr(IrSerializationFormat format) {
  (void)format;

  assertExecutableLoaded();

  if (executable_->isDeserialized()) {
    logging::session::warn(
        "Ir state is not populated when running from a serialized executable.");
    return std::string();
  }

  std::stringstream ss;
  ir.serialise(Ir::SerialiseFormat::JSON, ss);
  return ss.str();
}

void Session::initProgressLogger(const SessionOptions &userOptions) {
  if (userOptions.compilationProgressLogger) {
    int total = userOptions.compilationProgressTotal;
    userOptions.compilationProgressLogger(0, total);
  }
}

void Session::configureFromOnnx(const std::string &modelProtoOrFilename,
                                const DataFlow &df,
                                const TensorId &lossIn,
                                const Optimizer *optimizerIn,
                                const InputShapeInfo &perk,
                                std::shared_ptr<DeviceInfo> deviceInfo,
                                const SessionOptions &userOptions,
                                const Patterns &patterns) {
  POPART_TRACEPOINT();
  logging::session::trace("Session::configureFromOnnx");
  initProgressLogger(userOptions);

  auto modelProto = onnxutil::getModelProto(modelProtoOrFilename);

  if (userOptions.enableEngineCaching) {
    cacheEntries = getCacheEntries(userOptions.cachePath);
  }

  ir.prepare({modelProto,
              perk,
              df,
              lossIn,
              optimizerIn,
              *deviceInfo,
              userOptions,
              patterns},
             cacheEntries);
  setDevice(deviceInfo);
}

InferenceSession::~InferenceSession() = default;

std::unique_ptr<InferenceSession>
InferenceSession::createFromIr(Ir ir, std::shared_ptr<DeviceInfo> deviceInfo) {
  POPART_TRACEPOINT();
  logging::session::trace("InferenceSession::createFromIr");

  if (!deviceInfo) {
    throw error("InferenceSession::createFromIr: Must pass valid DeviceInfo.");
  }

  if (!ir.isPrepared()) {
    throw error("InferenceSession::createFromIr: Ir must be prepared.");
  }

  auto session = std::unique_ptr<InferenceSession>(
      new InferenceSession(std::move(ir), std::move(deviceInfo)));

  return session;
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

  const TensorId nullLoss        = {};
  const Optimizer *nullOptimizer = nullptr;

  auto session = std::unique_ptr<InferenceSession>(new InferenceSession());
  session->configureFromOnnx(model,
                             dataFlow,
                             nullLoss,
                             nullOptimizer,
                             inputShapeInfo,
                             std::move(deviceInfo),
                             userOptions,
                             patterns);

  return session;
}

TrainingSession::~TrainingSession() = default;

std::unique_ptr<TrainingSession>
TrainingSession::createFromIr(Ir ir, std::shared_ptr<DeviceInfo> deviceInfo) {
  POPART_TRACEPOINT();
  logging::session::trace("TrainingSession::createFromIr");

  if (!deviceInfo) {
    throw error("TrainingSession::createFromIr: Must pass valid DeviceInfo.");
  }

  if (!ir.isPrepared()) {
    throw error("TrainingSession::createFromIr: Ir must be prepared.");
  }

  auto session = std::unique_ptr<TrainingSession>(
      new TrainingSession(std::move(ir), std::move(deviceInfo)));

  return session;
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
                             &optimizer,
                             inputShapeInfo,
                             deviceInfo,
                             userOptions,
                             patterns);

  return session;
}

void TrainingSession::updateOptimizerFromHost(const Optimizer *optimizer) {
  POPART_TRACEPOINT();
  logging::session::trace("TrainingSession::updateOptimizerFromHost");

  assertExecutableLoaded();

  ir.updateOptimizer(*optimizer);
  executable_->updateOptimizerTensors();

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
