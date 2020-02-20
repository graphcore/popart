#ifndef GUARD_NEURALNET_POPDEVICEXMANAGER_HPP
#define GUARD_NEURALNET_POPDEVICEXMANAGER_HPP

#include <poplar/DeviceManager.hpp>

#include <popart/devicemanager.hpp>
#include <popart/popx/devicex.hpp>

namespace popart {
namespace popx {

class DevicexManager : public popart::DeviceProvider {
public:
  DevicexManager();

  virtual std::shared_ptr<DeviceInfo>
  getDevice(SyncPattern syncPattern, uint32_t deviceManagerId) override;

  virtual void
  enumerate(std::vector<std::shared_ptr<popart::DeviceInfo>> &devices,
            unsigned requiredNumIPUs,
            SyncPattern syncPattern,
            DeviceType type) override;

  virtual std::shared_ptr<popart::DeviceInfo>
  createHostDevice(popart::DeviceType type,
                   const std::map<std::string, std::string> &options) override;

private:
};

class DevicexInfo : public popart::DeviceInfo {
public:
  DevicexInfo(DeviceProvider &_provider,
              popart::DeviceType _type,
              poplar::Device &_device)
      : popart::DeviceInfo(_provider, _type), device(std::move(_device)) {}

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

  std::set<Devicex *> previouslyLoadedDevicexs;

protected:
  poplar::Device device;
};

class DevicexCpuInfo : public DevicexInfo {
public:
  DevicexCpuInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, popart::DeviceType::Cpu, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-cpu>"; }
};
class DevicexSimInfo : public DevicexInfo {
public:
  DevicexSimInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, popart::DeviceType::Sim, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-sim>"; }
};
class DevicexIpuModelInfo : public DevicexInfo {
public:
  DevicexIpuModelInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider, popart::DeviceType::IpuModel, _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-ipumodel>"; }
};
class DevicexIpuInfo : public DevicexInfo {
public:
  DevicexIpuInfo(DeviceProvider &_provider, int _id, poplar::Device &_device)
      : DevicexInfo(_provider, popart::DeviceType::Ipu, _device), id(_id) {}

  virtual bool attach();
  virtual void detach();

  virtual int getId() const { return id; }
  virtual std::string getVersion() const;

private:
  bool isAttached = false;
  int id;
};

popart::DeviceType convertDeviceType(poplar::TargetType targetType);
poplar::TargetType convertDeviceType(popart::DeviceType targetType);
void addSyncConfig(const SyncPattern syncPattern, poplar::OptionFlags &flags);

} // namespace popx
} // namespace popart

#endif
