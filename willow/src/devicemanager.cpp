#include <sstream>
#include <poponnx/devicemanager.hpp>
#include <poponnx/error.hpp>

namespace poponnx {

DeviceManager &DeviceManager::createDeviceManager() {
  static DeviceManager deviceManager;
  return deviceManager;
}

void DeviceManager::registerDeviceProvider(DeviceProvider *provider) {
  providers.push_back(provider);
}

std::vector<std::unique_ptr<DeviceInfo>> DeviceManager::enumerateDevices() {
  std::vector<std::unique_ptr<DeviceInfo>> devices;
  for (auto p : providers) {
    p->enumerate(devices);
  }
  return devices;
}

std::unique_ptr<DeviceInfo> DeviceManager::createCpuDevice() {
  for (auto p : providers) {
    std::unique_ptr<DeviceInfo> device =
        p->createHostDevice(DeviceType::Cpu, {});
    if (device != nullptr)
      return device;
  }
  return nullptr;
}

std::unique_ptr<DeviceInfo> DeviceManager::createIpuModelDevice(
    std::map<std::string, std::string> &options) {
  for (auto p : providers) {
    std::unique_ptr<DeviceInfo> device =
        p->createHostDevice(DeviceType::IpuModel, options);
    if (device != nullptr)
      return device;
  }
  return nullptr;
}

std::unique_ptr<DeviceInfo>
DeviceManager::createSimDevice(std::map<std::string, std::string> &options) {
  for (auto p : providers) {
    std::unique_ptr<DeviceInfo> device =
        p->createHostDevice(DeviceType::Sim, options);
    if (device != nullptr)
      return device;
  }
  return nullptr;
}

std::unique_ptr<DeviceInfo> DeviceManager::acquireAvaliableDevice() {

  auto devices = enumerateDevices();

  for (auto &device : devices) {
    if (device->attach())
      return std::move(device);
  }

  return nullptr;
}

std::unique_ptr<DeviceInfo>
DeviceManager::acquireAvaliableDevice(int numIpus, int tilesPerIpu) {

  auto devices = enumerateDevices();

  for (auto &device : devices) {
    if (numIpus == device->getNumIpus() &&
        tilesPerIpu == device->getTilesPerIpu()) {

      // Attach to the device. Will succeed if avaliable
      if (device->attach()) {
        return std::move(device);
      }
    }
  }

  return nullptr;
}

std::unique_ptr<DeviceInfo> DeviceManager::acquireDeviceById(int id) {

  auto devices = enumerateDevices();

  for (auto &device : devices) {
    if (device->getId() == id) {
      if (device->attach())
        return std::move(device);
    }
  }

  return nullptr;
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

std::string DeviceInfo::toString() const {
  std::stringstream ss;

  ss << "Device Type:" << getType() << " Id:" << getId()
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

} // namespace poponnx
