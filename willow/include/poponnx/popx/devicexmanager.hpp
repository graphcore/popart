#ifndef GUARD_NEURALNET_POPDEVICEXMANAGER_HPP
#define GUARD_NEURALNET_POPDEVICEXMANAGER_HPP

#include <poplar/DeviceManager.hpp>

#include <poponnx/devicemanager.hpp>

namespace willow {
namespace popx {

class DevicexManager : public willow::DeviceProvider {
public:
  DevicexManager();

  virtual void
  enumerate(std::vector<std::unique_ptr<willow::DeviceInfo>> &devices) override;

  virtual std::unique_ptr<willow::DeviceInfo>
  createHostDevice(willow::DeviceType type,
                   const std::map<std::string, std::string> &options);

private:
};

class DevicexInfo : public willow::DeviceInfo {
public:
  DevicexInfo(DeviceProvider &_provider,
              willow::DeviceType _type,
              poplar::Device &_device)
      : willow::DeviceInfo(_provider, _type), device(std::move(_device)) {}

  virtual bool attach();
  virtual void detach();

  void createVirtualGraph(int tilesPerIpu);

  virtual int getNumIpus() const { return device.getTarget().getNumIPUs(); }
  virtual int getTilesPerIpu() const {
    return device.getTarget().getTilesPerIPU();
  }
  virtual int getNumWorkerContexts() const {
    return device.getTarget().getNumWorkerContexts();
  }

  virtual std::vector<unsigned> getDriverIds() const {
    return device.getDriverIDs();
  }

  poplar::Device getDevice() { return std::move(device); }

protected:
  poplar::Device device;
};

class DevicexCpuInfo : public DevicexInfo {
public:
  DevicexCpuInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, willow::DeviceType::Cpu, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-cpu>"; }
};
class DevicexSimInfo : public DevicexInfo {
public:
  DevicexSimInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, willow::DeviceType::Sim, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-sim>"; }
};
class DevicexIpuModelInfo : public DevicexInfo {
public:
  DevicexIpuModelInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, willow::DeviceType::IpuModel, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-ipumodel>"; }
};
class DevicexIpuInfo : public DevicexInfo {
public:
  DevicexIpuInfo(DeviceProvider &_provider, int _id, poplar::Device &_device)
      : DevicexInfo(_provider, willow::DeviceType::Ipu, _device), id(_id) {}

  virtual bool attach();
  virtual void detach();

  virtual int getId() const { return id; }
  virtual std::string getVersion() const;

private:
  bool isAttached = false;
  int id;
};

} // namespace popx
} // namespace willow

#endif
