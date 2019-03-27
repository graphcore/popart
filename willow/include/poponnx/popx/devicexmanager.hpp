#ifndef GUARD_NEURALNET_POPDEVICEXMANAGER_HPP
#define GUARD_NEURALNET_POPDEVICEXMANAGER_HPP

#include <poplar/DeviceManager.hpp>

#include <poponnx/devicemanager.hpp>

namespace poponnx {
namespace popx {

class DevicexManager : public poponnx::DeviceProvider {
public:
  DevicexManager();

  virtual void enumerate(
      std::vector<std::shared_ptr<poponnx::DeviceInfo>> &devices) override;

  virtual std::shared_ptr<poponnx::DeviceInfo>
  createHostDevice(poponnx::DeviceType type,
                   const std::map<std::string, std::string> &options) override;

private:
};

class DevicexInfo : public poponnx::DeviceInfo {
public:
  DevicexInfo(DeviceProvider &_provider,
              poponnx::DeviceType _type,
              poplar::Device &_device)
      : poponnx::DeviceInfo(_provider, _type), device(std::move(_device)) {}

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

  poplar::Device &getDevice() { return device; }

protected:
  poplar::Device device;
};

class DevicexCpuInfo : public DevicexInfo {
public:
  DevicexCpuInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, poponnx::DeviceType::Cpu, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-cpu>"; }
};
class DevicexSimInfo : public DevicexInfo {
public:
  DevicexSimInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, poponnx::DeviceType::Sim, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-sim>"; }
};
class DevicexIpuModelInfo : public DevicexInfo {
public:
  DevicexIpuModelInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, poponnx::DeviceType::IpuModel, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-ipumodel>"; }
};
class DevicexIpuInfo : public DevicexInfo {
public:
  DevicexIpuInfo(DeviceProvider &_provider, int _id, poplar::Device &_device)
      : DevicexInfo(_provider, poponnx::DeviceType::Ipu, _device), id(_id) {}

  virtual bool attach();
  virtual void detach();

  virtual int getId() const { return id; }
  virtual std::string getVersion() const;

private:
  bool isAttached = false;
  int id;
};

poponnx::DeviceType convertDeviceType(poplar::TargetType targetType);

} // namespace popx
} // namespace poponnx

#endif
