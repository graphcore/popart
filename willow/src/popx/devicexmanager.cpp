
#include <poponnx/popx/devicexmanager.hpp>

#include <poponnx/error.hpp>
#include <poponnx/util.hpp>

#include <poplar/Device.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/exceptions.hpp>

#include <algorithm>

namespace willow {
namespace popx {

willow::DeviceType convertDeviceType(poplar::TargetType targetType) {
  switch (targetType) {
  case poplar::TargetType::IPU:
    return DeviceType::Ipu;
  case poplar::TargetType::IPU_MODEL:
    return DeviceType::IpuModel;
  case poplar::TargetType::CPU:
    return DeviceType::Cpu;
  };
  throw error("Unknonw target type");
}

DevicexManager::DevicexManager() {
  DeviceManager::getDeviceManager().registerDeviceProvider(this);
}

void DevicexManager::enumerate(
    std::vector<std::unique_ptr<willow::DeviceInfo>> &devices) {

  auto deviceManager = poplar::DeviceManager::getDeviceManager();
  std::vector<poplar::Device> popdevices = deviceManager.getDevices();

  for (auto &device : popdevices) {

    willow::DeviceType type =
        convertDeviceType(device.getTarget().getTargetType());
    switch (type) {
    case DeviceType::Ipu: {
      std::unique_ptr<willow::DeviceInfo> ipu =
          make_unique<DevicexIpuInfo>(*this, device.getId(), device);
      devices.push_back(std::move(ipu));
    } break;
    default: {
      // Do nothing for the 'host' devices
    }
    }
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
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) -> unsigned char { return std::tolower(c); });
    return ((it->second == "true") ? true : false);
  } else {
    return defaultValue;
  }
}

std::unique_ptr<willow::DeviceInfo> DevicexManager::createHostDevice(
    willow::DeviceType type,
    const std::map<std::string, std::string> &options) {

  switch (type) {
  case DeviceType::Cpu: {
    poplar::Device device = poplar::Device::createCPUDevice();
    return make_unique<DevicexCpuInfo>(*this, device);
  }
  case DeviceType::IpuModel: {

    // Create an ipumodel, using the values set in the options map, else use the
    // defaults
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs        = mapFind(options, "numIPUs", 1);
    ipuModel.tilesPerIPU    = mapFind(options, "tilesPerIPU", 1216);
    ipuModel.compileIPUCode = mapFind(options, "compileIPUCode", true);

    poplar::Device device = ipuModel.createDevice();
    return make_unique<DevicexIpuModelInfo>(*this, device);
  }
  case DeviceType::Sim: {
    try {

      // Use the values from the options map or use the defaults.
      int numIPUs     = mapFind(options, "numIPUs", 1);
      int tilesPerIPU = mapFind(options, "tilesPerIPU", 20);

      auto target =
          poplar::Target::createIPUTarget(numIPUs, tilesPerIPU, "_TEST_SYSTEM");

      poplar::Device device = poplar::Device::createSimulatorDevice(target);
      return make_unique<DevicexSimInfo>(*this, device);
    } catch (const poplar::poplar_error &e) {
      throw error("Simulator not supported. " + std::string(e.what()));
    }
  }
  case DeviceType::Ipu: {
    throw error("Can not create device of type ipu");
  }
  }

  // return null if we can not create a device that meets the requirements
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
DevicexManager s_devicexManager;
}

} // namespace popx
} // namespace willow