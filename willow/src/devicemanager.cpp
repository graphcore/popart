// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cxxabi.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <poplar/OptionFlags.hpp>
#include <poplar/StringRef.hpp>
#include <poplar/Target.hpp>
#include <poplar/exceptions.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/util.hpp>

#include "popart/logging.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {

SyncPattern syncPatternFromString(const std::string &str) {
  if (str == "full") {
    return SyncPattern::Full;
  }
  if (str == "singlePipeline") {
    return SyncPattern::SinglePipeline;
  }
  if (str == "replicaAndLadder") {
    return SyncPattern::ReplicaAndLadder;
  }

  throw error("Unknown syncPattern setting: {}", str);
}

std::string syncPatternToString(const SyncPattern &pattern) {
  switch (pattern) {
  case SyncPattern::Full:
    return "full";
  case SyncPattern::SinglePipeline:
    return "singlePipeline";
  case SyncPattern::ReplicaAndLadder:
    return "replicaAndLadder";
  }
  throw error("Unknown syncPattern setting: {}", static_cast<int>(pattern));
}

DeviceManager &DeviceManager::createDeviceManager() {
  static DeviceManager deviceManager;
  return deviceManager;
}

void DeviceManager::registerDeviceProvider(DeviceProvider *provider) {
  providers.push_back(provider);
}

void DeviceManager::enumerate(
    std::vector<std::shared_ptr<popart::DeviceInfo>> &devices,
    unsigned requiredNumIPUs,
    SyncPattern syncPattern,
    DeviceType type,
    DeviceConnectionType connectionType,
    uint32_t requiredTilesPerIPU) {
  for (auto provider : providers) {
    provider->enumerate(devices,
                        requiredNumIPUs,
                        syncPattern,
                        type,
                        connectionType,
                        requiredTilesPerIPU);
  }
}

std::shared_ptr<DeviceInfo>
DeviceManager::getDevice(SyncPattern syncPattern,
                         unsigned deviceManagerId,
                         DeviceConnectionType connectionType) {
  if (connectionType == DeviceConnectionType::Never) {
    throw error("Trying to acquire a hardware device when connectionType is "
                "DeviceConnectionType::Never. For offline compilation, use "
                "createOfflineIPUDevice");
  }
  for (auto p : providers) {
    auto device = p->getDevice(syncPattern, deviceManagerId, connectionType);
    if (device != nullptr) {
      return device;
    }
  }
  return nullptr;
}

std::vector<std::shared_ptr<DeviceInfo>>
DeviceManager::enumerateDevices(SyncPattern pattern,
                                int numIpus,
                                DeviceType deviceType,
                                DeviceConnectionType connectionType,
                                int tilesPerIPU) {
  std::vector<std::shared_ptr<DeviceInfo>> devices;

  for (auto p : providers) {
    p->enumerate(
        devices, numIpus, pattern, deviceType, connectionType, tilesPerIPU);
  }
  for (auto d : devices) {
    logging::debug("Device: {}", d.get()->toString());
  }

  for (auto device : devices) {
    device->setOnDemandAttachTimeout(attachTimeout);
  }

  return devices;
}

std::shared_ptr<DeviceInfo> DeviceManager::createHostDevice(
    DeviceType type,
    const std::map<std::string, std::string> &options) {
  for (auto p : providers) {
    std::shared_ptr<DeviceInfo> device = p->createHostDevice(type, options);
    if (device != nullptr) {
      return device;
    }
  }

  // Unable to create host device
  std::vector<std::string> opts;
  opts.reserve(options.size());
  for (auto opt : options) {
    std::stringstream ss;
    ss << opt.first << ": " << opt.second;
    opts.push_back(ss.str());
  }
  throw error("Could not acquire {} device with options [{}] from any of {} "
              "providers.",
              type,
              logging::join(opts.begin(), opts.end(), ","),
              providers.size());
}

std::shared_ptr<DeviceInfo> DeviceManager::createCpuDevice() {
  return createHostDevice(DeviceType::Cpu, {});
}

std::shared_ptr<DeviceInfo> DeviceManager::createIpuModelDevice(
    std::map<std::string, std::string> &options) {
  return createHostDevice(DeviceType::IpuModel, options);
}

std::shared_ptr<DeviceInfo>
DeviceManager::createSimDevice(std::map<std::string, std::string> &options) {
  return createHostDevice(DeviceType::Sim, options);
}

std::shared_ptr<DeviceInfo> DeviceManager::createOfflineIPUDevice(
    std::map<std::string, std::string> &options) {
  return createHostDevice(DeviceType::OfflineIpu, options);
}

std::shared_ptr<DeviceInfo>
DeviceManager::createOfflineIpuFromDeviceInfo(const DeviceInfo &deviceInfo) {
  for (auto p : providers) {
    return p->createOfflineIpuFromDeviceInfo(deviceInfo);
  }
  throw internal_error("Could not find a provider to construct offline IPU.");
}

std::shared_ptr<DeviceInfo> DeviceManager::tryAcquireAvailableDevice(
    int numIpus,
    int tilesPerIPU,
    SyncPattern pattern,
    DeviceConnectionType connectionType,
    DeviceSelectionCriterion selectionCriterion) {
  if (numIpus > 0 && ((numIpus & (numIpus - 1)) != 0)) {
    throw error("You have attempted to acquire {} IPUs. The number of IPUs "
                "requested must be a power of two",
                numIpus);
  }
  if (connectionType == DeviceConnectionType::Never) {
    throw error("Trying to acquire a hardware device when connectionType is "
                "DeviceConnectionType::Never");
  }

  auto devices = enumerateDevices(
      pattern, numIpus, DeviceType::Ipu, connectionType, tilesPerIPU);

  std::mt19937 g(/* seed */ 1);

  if (selectionCriterion == DeviceSelectionCriterion::Random) {
    std::shuffle(devices.begin(), devices.end(), g);
  }

  for (auto &device : devices) {
    if ((!tilesPerIPU || tilesPerIPU == device->getTilesPerIPU())) {
      // Attach to the device. Will succeed if available
      if (connectionType == DeviceConnectionType::Always) {
        if (device->attach()) {
          return device;
        }
      } else {
        device->writeToDeviceAccessLog("select");
        return device;
      }
    }
  }

  // Return nullptr if no device is acquired.
  return nullptr;
}

std::shared_ptr<DeviceInfo> DeviceManager::acquireAvailableDevice(
    int numIpus,
    int tilesPerIPU,
    SyncPattern pattern,
    DeviceConnectionType connectionType,
    DeviceSelectionCriterion selectionCriterion) {
  auto device = tryAcquireAvailableDevice(
      numIpus, tilesPerIPU, pattern, connectionType, selectionCriterion);

  if (!device) {
    throw error(
        "Failed to acquire device with {} IPUs. Ensure that there are "
        "sufficient IPUs available. If you have enabled the Poplar SDK you can "
        "check device availability with the `gc-monitor` command-line utility.",
        numIpus);
  }
  return device;
}

std::shared_ptr<DeviceInfo>
DeviceManager::tryAcquireDeviceById(int id,
                                    SyncPattern pattern,
                                    DeviceConnectionType connectionType) {
  if (connectionType == DeviceConnectionType::Never) {
    throw error("Trying to acquire a hardware device when connectionType is "
                "DeviceConnectionType::Never");
  }

  auto device = getDevice(pattern, id, connectionType);
  device->setOnDemandAttachTimeout(attachTimeout);

  // Attach to the device. Will succeed if available
  if (connectionType == DeviceConnectionType::Always) {
    if (device->attach()) {
      return device;
    } else {
      // Return nullptr if no device is acquired.
      return nullptr;
    }
  }
  return device;
}

std::shared_ptr<DeviceInfo>
DeviceManager::acquireDeviceById(int id,
                                 SyncPattern pattern,
                                 DeviceConnectionType connectionType) {
  auto device = tryAcquireDeviceById(id, pattern, connectionType);

  // Warn if acquiring device is unsuccessful. TODO T46787: error instead.
  if (!device) {
    throw error(
        "Failed to acquire device with id '{}'. Ensure it is available. If you "
        "have enabled the Poplar SDK you can check device availability with "
        "the `gc-monitor` command-line utility.",
        id);
  }
  return device;
}

void DeviceManager::setOnDemandAttachTimeout(const unsigned seconds) {
  attachTimeout = seconds;
}

std::ostream &operator<<(std::ostream &os, const DeviceType &dt) {
  switch (dt) {
  case DeviceType::Cpu:
    os << "cpu";
    break;
  case DeviceType::Ipu:
    os << "ipu";
    break;
  case DeviceType::IpuModel:
    os << "ipu-model";
    break;
  case DeviceType::OfflineIpu:
    os << "offline-ipu";
    break;
  case DeviceType::Sim:
    os << "sim";
    break;
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, const DeviceConnectionType &dct) {
  switch (dct) {
  case DeviceConnectionType::Always:
    os << "always";
    break;
  case DeviceConnectionType::OnDemand:
    os << "on-demand";
    break;
  case DeviceConnectionType::Never:
    os << "never";
    break;
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, const SyncPattern &sp) {
  os << syncPatternToString(sp);
  return os;
}

DeviceInfo::DeviceInfo(DeviceProvider &_provider,
                       DeviceType _type,
                       DeviceConnectionType _connectionType,
                       const poplar::OptionFlags &_flags)
    : provider(_provider), type(_type), connectionType(_connectionType),
      flags(std::make_unique<const poplar::OptionFlags>(_flags)) {
  (void)provider;
}

DeviceInfo::~DeviceInfo() {}

bool DeviceInfo::isHwCompatible() const {
  return type == popart::DeviceType::Ipu ||
         type == popart::DeviceType::OfflineIpu;
}

const poplar::OptionFlags &DeviceInfo::getOptionFlags() const { return *flags; }

std::string DeviceInfo::toString() const {
  std::stringstream ss;

  ss << "Device Type:" << getType()
     << " Connection Type:" << getConnectionType() << " Id:" << getId()
     << " Version:" << getVersion() << " NumIPUs:" << getNumIpus()
     << " NumTilesPerIPU:" << getTilesPerIPU();

  ss << " DeviceIds: {";
  std::string sep;
  for (unsigned i : getDriverIds()) {
    ss << sep << i;
    sep = ",";
  }
  ss << "}";

  // TODO : Add all the information from Target

  return ss.str();
}

void DeviceInfo::setOnDemandAttachTimeout(const unsigned seconds) {
  attachTimeout = seconds;
}

bool DeviceInfo::tryAttachUntilTimeout() {

  auto start = std::chrono::steady_clock::now();

  writeToDeviceAccessLog(
      "try-attach-start",
      {{"timeout", std::to_string(getOnDemandAttachTimeout())}});

  // Periodically try to attach until either timeout reached or
  // successfully attached
  auto startTime = std::chrono::steady_clock::now();
  unsigned wait  = 0;
  bool attached  = false;
  while (wait < getOnDemandAttachTimeout() && !attached) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    attached       = attach();
    auto delayTime = std::chrono::steady_clock::now();
    wait =
        std::chrono::duration_cast<std::chrono::seconds>(delayTime - startTime)
            .count();
    writeToDeviceAccessLog(
        "try-attach-wait",
        {{"wait", std::to_string(wait)},
         {"timeout", std::to_string(getOnDemandAttachTimeout())}});
  }

  auto end = std::chrono::steady_clock::now();
  auto total =
      std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

  if (attached) {
    writeToDeviceAccessLog(
        "try-attach-success",
        {{"try-time", std::to_string(total)},
         {"timeout", std::to_string(getOnDemandAttachTimeout())}});
  } else {
    writeToDeviceAccessLog(
        "try-attach-fail",
        {{"try-time", std::to_string(total)},
         {"timeout", std::to_string(getOnDemandAttachTimeout())}});
  }

  return attached;
}

void DeviceInfo::writeToDeviceAccessLog(
    const std::string &event,
    const std::map<std::string, std::string> &auxKeyVals) {

  auto deviceLog = getPopartEnvVar("LOG_DEVICE_ACCESS_IN_TESTS");

  if (deviceLog) {

    if (deviceAccessLogEntryFmt.empty()) {
      // NOTE: This string is cached because constructing it involves calls to
      // virtual functions. Calling virtual functions from the destructor causes
      // run-time issues, so by caching the string we can still log the destruct
      // in the constructor.
      std::stringstream ss;

      auto testName = getPopartEnvVar("TEST_NAME");
      if (testName) {
        ss << ", test:" << *testName;

        // Try and add pytest test name if there is one. This environment
        // variable is set by pytest (see, e.g.,
        // https://docs.pytest.org/en/6.2.x/example/simple.html#pytest-current-test-environment-variable).
        auto pytestTestName = getEnvVar("PYTEST_CURRENT_TEST");
        if (pytestTestName) {
          // Use name from pytest if there is one.
          auto slash = pytestTestName->find_last_of('/');
          auto space = pytestTestName->find_last_of(' ');

          if (slash != std::string::npos && space != std::string::npos) {
            // Format "/path/to/test_foo.py::test1 (call)" as
            // "test_foo.py::test1"
            ss << "(" << pytestTestName->substr(slash + 1, space - slash - 1)
               << ")";
          } else {
            // Don't format.
            ss << "(" << *pytestTestName << ")";
          }
        }
      }

      ss << ", event:{}";
      ss << ", ipus:"
         << "[";
      auto childIds = getChildIds();
      ss << logging::join(childIds.begin(), childIds.end(), ",");
      ss << "]";

      deviceAccessLogEntryFmt = ss.str();
    }

    auto now       = std::chrono::system_clock::now();
    auto nowTime   = std::chrono::system_clock::to_time_t(now);
    auto localTime = std::localtime(&nowTime);

    std::ofstream outfile;
    outfile.open(*deviceLog, std::ios_base::app);
    outfile << std::put_time(localTime, "%FT%T") << "Z";
    outfile << logging::format(deviceAccessLogEntryFmt, event);

    for (const auto &entry : auxKeyVals) {
      outfile << ", " << entry.first << ":" << entry.second;
    }

    outfile << "\n";
    outfile.close();
  }
}

std::ostream &operator<<(std::ostream &os, const DeviceInfo &di) {
  return os << di.toString();
}

} // namespace popart

namespace std {
std::size_t
std::hash<popart::DeviceInfo>::operator()(const popart::DeviceInfo &di) const {
  std::size_t seed = 0;

  boost::hash_combine(seed, di.isHwCompatible());

  poplar::StringRef targetArchString;
  try {
    // Some devices don't implement getTargetArchString()
    targetArchString = di.getTarget().getTargetArchString();
  } catch (const poplar::poplar_error &) {
  }

  if (targetArchString.empty()) {
    const auto &options = di.getOptionFlags();
    try {
      targetArchString = options.at("ipuVersion");
    } catch (const std::out_of_range &) {
    }
  }

  if (!targetArchString.empty()) {
    boost::hash_combine(seed, std::string{targetArchString});
  }
  boost::hash_combine(seed, di.getNumIpus());
  boost::hash_combine(seed, di.getTilesPerIPU());
  boost::hash_combine(seed, di.getNumWorkerContexts());

  return seed;
}
} // namespace std
