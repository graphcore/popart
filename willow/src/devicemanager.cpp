// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <sstream>
#include <poplar/DeviceManager.hpp>
#include <poplar/OptionFlags.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>

namespace popart {

DeviceManager &DeviceManager::createDeviceManager() {
  static DeviceManager deviceManager;
  return deviceManager;
}

void DeviceManager::registerDeviceProvider(DeviceProvider *provider) {
  providers.push_back(provider);
}

std::shared_ptr<DeviceInfo>
DeviceManager::getDevice(SyncPattern syncPattern,
                         unsigned deviceManagerId,
                         DeviceConnectionType connectionType) {
  if (connectionType == DeviceConnectionType::NEVER) {
    throw error("Trying to acquire a hardware device when connectionType is "
                "DeviceConnectionType::NEVER");
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
                                uint32_t /*replication_factor*/,
                                int numIpus,
                                DeviceType deviceType,
                                DeviceConnectionType connectionType) {
  std::vector<std::shared_ptr<DeviceInfo>> devices;

  for (auto p : providers) {
    p->enumerate(devices, numIpus, pattern, deviceType, connectionType);
  }
  for (auto d : devices) {
    logging::debug("Device: {}", d.get()->toString());
  }
  return devices;
}

std::shared_ptr<DeviceInfo> DeviceManager::createCpuDevice() {
  for (auto p : providers) {
    std::shared_ptr<DeviceInfo> device =
        p->createHostDevice(DeviceType::Cpu, {});
    if (device != nullptr)
      return device;
  }
  return nullptr;
}

std::shared_ptr<DeviceInfo> DeviceManager::createIpuModelDevice(
    std::map<std::string, std::string> &options) {
  for (auto p : providers) {
    std::shared_ptr<DeviceInfo> device =
        p->createHostDevice(DeviceType::IpuModel, options);
    if (device != nullptr)
      return device;
  }
  return nullptr;
}

std::shared_ptr<DeviceInfo>
DeviceManager::createSimDevice(std::map<std::string, std::string> &options) {
  for (auto p : providers) {
    std::shared_ptr<DeviceInfo> device =
        p->createHostDevice(DeviceType::Sim, options);
    if (device != nullptr)
      return device;
  }
  return nullptr;
}

std::shared_ptr<DeviceInfo>
DeviceManager::acquireAvailableDevice(int numIpus,
                                      int tilesPerIpu,
                                      SyncPattern pattern,
                                      uint32_t replication_factor,
                                      DeviceConnectionType connectionType) {
  if (replication_factor > 1) {
    logging::devicex::warn(
        "You have specified a replication_factor in the call to aquire "
        "available devices. This parameter is deprecated and will have no "
        "effect. Please account for the replication factor when calculating "
        "the number of IPUs required.");
  }
  if (numIpus > 0 && ((numIpus & (numIpus - 1)) != 0)) {
    throw error("You have attempted to acquire {} IPUs. The number of IPUs "
                "requested must be a power of two",
                numIpus);
  }
  if (connectionType == DeviceConnectionType::NEVER) {
    throw error("Trying to acquire a hardware device when connectionType is "
                "DeviceConnectionType::NEVER");
  }

  auto devices = enumerateDevices(
      pattern, replication_factor, numIpus, DeviceType::Ipu, connectionType);

  for (auto &device : devices) {
    if ((!tilesPerIpu || tilesPerIpu == device->getTilesPerIpu())) {
      // Attach to the device. Will succeed if available
      if (connectionType == DeviceConnectionType::ALWAYS) {
        if (device->attach()) {
          return device;
        }
      } else {
        return device;
      }
    }
  }

  return nullptr;
}

std::shared_ptr<DeviceInfo>
DeviceManager::acquireDeviceById(int id,
                                 SyncPattern pattern,
                                 uint32_t replication_factor,
                                 DeviceConnectionType connectionType) {
  if (connectionType == DeviceConnectionType::NEVER) {
    throw error("Trying to acquire a hardware device when connectionType is "
                "DeviceConnectionType::NEVER");
  }

  auto device = getDevice(pattern, id, connectionType);

  if (replication_factor > 1) {
    logging::devicex::warn(
        "You have specified a replication_factor in the call to aquire "
        "available devices. This parameter is deprecated and will have no "
        "effect. Please account for the replication factor when calculating "
        "the number of IPUs required.");
  }

  // Attach to the device. Will succeed if available
  if (connectionType == DeviceConnectionType::ALWAYS) {
    if (device->attach()) {
      return device;
    } else {
      return nullptr;
    }
  }

  return device;
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
  case DeviceType::Sim:
    os << "sim";
    break;
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, const DeviceConnectionType &dct) {
  switch (dct) {
  case DeviceConnectionType::ALWAYS:
    os << "always";
    break;
  case DeviceConnectionType::ON_DEMAND:
    os << "on-demand";
    break;
  case DeviceConnectionType::NEVER:
    os << "never";
    break;
  }

  return os;
}

std::string DeviceInfo::toString() const {
  std::stringstream ss;

  ss << "Device Type:" << getType()
     << " Connection Type:" << getConnectionType() << " Id:" << getId()
     << " Version:" << getVersion() << " NumIPUs:" << getNumIpus()
     << " NumTilesPerIPU:" << getTilesPerIpu();

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

std::ostream &operator<<(std::ostream &os, const DeviceInfo &di) {
  return os << di.toString();
}

} // namespace popart
