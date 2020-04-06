// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <boost/range/algorithm/find.hpp>

#include <popart/popx/devicexmanager.hpp>

#include <memory>
#include <popart/error.hpp>
#include <popart/util.hpp>

#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/exceptions.hpp>

#include <algorithm>
#include <sstream>

using boost::find;

namespace popart {
namespace popx {

popart::DeviceType convertDeviceType(poplar::TargetType targetType) {
  switch (targetType) {
  case poplar::TargetType::IPU:
    return DeviceType::Ipu;
  case poplar::TargetType::IPU_MODEL:
    return DeviceType::IpuModel;
  case poplar::TargetType::CPU:
    return DeviceType::Cpu;
  }
  throw error("Unknown target type");
}

poplar::TargetType convertDeviceType(popart::DeviceType deviceType) {
  switch (deviceType) {
  case DeviceType::Ipu:
    return poplar::TargetType::IPU;
  case DeviceType::IpuModel:
    return poplar::TargetType::IPU_MODEL;
  case DeviceType::Cpu:
    return poplar::TargetType::CPU;
  case DeviceType::Sim: {
    throw error("Sim not supported in convertDeviceType");
  }
  }
  throw error("Unknown device type");
}

void addSyncConfig(const SyncPattern syncPattern, poplar::OptionFlags &flags) {
  switch (syncPattern) {
  case SyncPattern::Full:
    break;
  case SyncPattern::SinglePipeline:
    flags.set("syncConfiguration", "ipuAndAll");
    break;
  case SyncPattern::PingPong:
    flags.set("syncConfiguration", "IntraReplicaAndLadder");
    break;
  }
}

DevicexManager::DevicexManager() {
  DeviceManager::createDeviceManager().registerDeviceProvider(this);
}

std::shared_ptr<DeviceInfo>
DevicexManager::getDevice(SyncPattern syncPattern, uint32_t deviceManagerId) {
  auto deviceManager = poplar::DeviceManager::createDeviceManager();

  poplar::OptionFlags flags;
  addSyncConfig(syncPattern, flags);
  auto device = deviceManager.getDevice(deviceManagerId, flags);
  return std::make_shared<DevicexIpuInfo>(*this, device.getId(), device);
}

void DevicexManager::enumerate(
    std::vector<std::shared_ptr<popart::DeviceInfo>> &devices,
    unsigned requiredNumIPUs,
    SyncPattern syncPattern,
    DeviceType type) {

  auto deviceManager = poplar::DeviceManager::createDeviceManager();

  poplar::OptionFlags flags;
  addSyncConfig(syncPattern, flags);
  std::vector<poplar::Device> popdevices =
      deviceManager.getDevices(convertDeviceType(type), requiredNumIPUs, flags);
  for (auto &device : popdevices) {
    std::shared_ptr<popart::DeviceInfo> ipu =
        std::make_shared<DevicexIpuInfo>(*this, device.getId(), device);
    devices.push_back(ipu);
  }
}

template <typename T>
T mapFind(const std::map<std::string, std::string> &map,
          const char *key,
          T defaultValue);

template <>
int mapFind<int>(const std::map<std::string, std::string> &map,
                 const char *key,
                 int defaultValue) {
  auto it = map.find(key);
  if (it != map.end()) {
    return std::stoi(it->second);
  } else {
    return defaultValue;
  }
}

template <>
bool mapFind<bool>(const std::map<std::string, std::string> &map,
                   const char *key,
                   bool defaultValue) {
  auto it = map.find(key);
  if (it != map.end()) {
    // make the value lower case
    std::string value = it->second;
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) -> unsigned char {
                     return static_cast<unsigned char>(std::tolower(c));
                   });
    return ((value == "true") ? true : false);
  } else {
    return defaultValue;
  }
}

template <>
std::string mapFind<std::string>(const std::map<std::string, std::string> &map,
                                 const char *key,
                                 std::string defaultValue) {
  auto it = map.find(key);
  if (it != map.end()) {
    return it->second;
  } else {
    return defaultValue;
  }
}

std::shared_ptr<popart::DeviceInfo> DevicexManager::createHostDevice(
    popart::DeviceType type,
    const std::map<std::string, std::string> &options) {

  auto checkOptions = [&](const std::vector<std::string> &validOptionKeys) {
    for (auto &key_value : options) {
      auto &key = key_value.first;
      if (find(validOptionKeys, key) == validOptionKeys.end()) {
        throw error("Invalid option `{}' passed to "
                    "DevicexManager::createHostDevice for device type {}. "
                    "Valid options are {}",
                    key,
                    type,
                    validOptionKeys);
      }
    }
  };

  switch (type) {
  case DeviceType::Cpu: {
    checkOptions({});
    poplar::Device device = poplar::Device::createCPUDevice();
    return std::make_shared<DevicexCpuInfo>(*this, device);
  }
  case DeviceType::IpuModel: {
    checkOptions({"numIPUs", "tilesPerIPU", "compileIPUCode", "ipuVersion"});

    // Create an ipumodel, using the values set in the options map, else use the
    // defaults
    const std::string ipuVersion =
        mapFind<std::string>(options, "ipuVersion", "ipu1");
    poplar::IPUModel ipuModel(ipuVersion.c_str());
    ipuModel.numIPUs = mapFind(options, "numIPUs", int(ipuModel.numIPUs));
    ipuModel.tilesPerIPU =
        mapFind(options, "tilesPerIPU", int(ipuModel.tilesPerIPU));
    ipuModel.compileIPUCode = mapFind(options, "compileIPUCode", true);

    poplar::Device device = ipuModel.createDevice();
    return std::make_shared<DevicexIpuModelInfo>(*this, device);
  }
  case DeviceType::Sim: {
    checkOptions({"numIPUs", "tilesPerIPU"});
    try {

      // Use the values from the options map or use the defaults.
      int numIPUs     = mapFind(options, "numIPUs", 1);
      int tilesPerIPU = mapFind(options, "tilesPerIPU", 20);

      auto target = poplar::Target::createIPUTarget(
          numIPUs, tilesPerIPU, "_VIRTUAL_GRAPH_TEST_16");
      poplar::Device device = poplar::Device::createSimulatorDevice(target);
      return std::make_shared<DevicexSimInfo>(*this, device);
    } catch (const poplar::poplar_error &e) {
      throw error("Simulator not supported. " + std::string(e.what()));
    }
  }
  case DeviceType::Ipu: {
    throw error("Cannot create device of type ipu");
  }
  }

  // return null if we cannot create a device that meets the requirements
  return nullptr;
}

bool DevicexInfo::attach() { return device.attach(); }

void DevicexInfo::detach() { device.detach(); }

void DevicexInfo::createVirtualGraph(int tilesPerIpu) {
  device = device.createVirtualDevice(tilesPerIpu);
}

bool DevicexIpuInfo::attach() {
  isAttached = true;
  return DevicexInfo::attach();
}

void DevicexIpuInfo::detach() {
  isAttached = false;
  DevicexInfo::detach();
}

std::string DevicexIpuInfo::getVersion() const {

  if (isAttached) {
    unsigned major, minor, point;
    device.getDriverVersion(major, minor, point);

    std::stringstream ss;
    ss << major << "." << minor << "." << point;
    return ss.str();
  } else {
    // If the device is not attached we do not know the version
    return "<unknown-ipu-unattached>";
  }
}

namespace {
static DevicexManager s_devicexManager;
}

} // namespace popx
} // namespace popart
