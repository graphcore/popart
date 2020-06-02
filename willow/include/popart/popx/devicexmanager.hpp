// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPDEVICEXMANAGER_HPP
#define GUARD_NEURALNET_POPDEVICEXMANAGER_HPP

#include <poplar/DeviceManager.hpp>

#include <popart/devicemanager.hpp>
#include <popart/error.hpp>
#include <popart/popx/devicex.hpp>

namespace popart {
namespace popx {

class DevicexManager : public popart::DeviceProvider {
public:
  DevicexManager();

  virtual std::shared_ptr<DeviceInfo>
  getDevice(SyncPattern syncPattern,
            uint32_t deviceManagerId,
            DeviceConnectionType connectionType) override;

  virtual void
  enumerate(std::vector<std::shared_ptr<popart::DeviceInfo>> &devices,
            unsigned requiredNumIPUs,
            SyncPattern syncPattern,
            DeviceType type,
            DeviceConnectionType connectionType) override;

  virtual std::shared_ptr<popart::DeviceInfo>
  createHostDevice(popart::DeviceType type,
                   const std::map<std::string, std::string> &options,
                   SyncPattern syncPattern = SyncPattern::Full) override;

private:
};

class DevicexInfo : public popart::DeviceInfo {
public:
  DevicexInfo(DeviceProvider &_provider,
              popart::DeviceType _type,
              popart::DeviceConnectionType _connectionType,
              poplar::Device &_device)
      : popart::DeviceInfo(_provider, _type, _connectionType),
        device(std::move(_device)) {}

  virtual bool attach();
  virtual void detach();

  virtual int getNumIpus() const { return getTarget().getNumIPUs(); }
  virtual int getTilesPerIpu() const { return getTarget().getTilesPerIPU(); }
  virtual int getNumWorkerContexts() const {
    return getTarget().getNumWorkerContexts();
  }

  virtual std::vector<unsigned> getDriverIds() const {
    return device.getDriverIDs();
  }

  poplar::Device &getDevice() { return device; }

  virtual const poplar::Target &getTarget() const {
    return device.getTarget();
  };

  std::set<Devicex *> previouslyLoadedDevicexs;

protected:
  poplar::Device device;
};

class DevicexCpuInfo : public DevicexInfo {
public:
  DevicexCpuInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider,
                    popart::DeviceType::Cpu,
                    popart::DeviceConnectionType::Always,
                    _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-cpu>"; }
};
class DevicexSimInfo : public DevicexInfo {
public:
  DevicexSimInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider,
                    popart::DeviceType::Sim,
                    popart::DeviceConnectionType::Always,
                    _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-sim>"; }
};
class DevicexIpuModelInfo : public DevicexInfo {
public:
  DevicexIpuModelInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider,
                    popart::DeviceType::IpuModel,
                    popart::DeviceConnectionType::Always,
                    _device) {}

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<unknown-ipumodel>"; }
};
class DevicexIpuInfo : public DevicexInfo {
public:
  DevicexIpuInfo(DeviceProvider &_provider,
                 popart::DeviceConnectionType _dct,
                 int _id,
                 poplar::Device &_device)
      : DevicexInfo(_provider, popart::DeviceType::Ipu, _dct, _device),
        id(_id) {}

  virtual bool attach();
  virtual void detach();

  virtual int getId() const { return id; }
  virtual std::string getVersion() const;

  virtual bool canCompileOffline() const { return true; }

private:
  bool isAttached = false;
  int id;
};

class DevicexOfflineIpuInfo : public popart::DeviceInfo {
public:
  DevicexOfflineIpuInfo(DeviceProvider &_provider, poplar::Target &_target)
      : popart::DeviceInfo(_provider,
                           popart::DeviceType::OfflineIpu,
                           popart::DeviceConnectionType::Never),
        target(std::move(_target)) {}

  virtual bool attach() { throw error("Cannot attach virtual device"); }

  virtual void detach() { throw error("Cannot detach virtual device"); }

  virtual int getId() const { return 0; }
  virtual std::string getVersion() const { return "<offline-ipu>"; }

  virtual int getNumIpus() const { return target.getNumIPUs(); }
  virtual int getTilesPerIpu() const { return target.getTilesPerIPU(); }
  virtual int getNumWorkerContexts() const {
    return target.getNumWorkerContexts();
  }

  virtual std::vector<unsigned> getDriverIds() const { return {0}; }

  virtual const poplar::Target &getTarget() const { return target; };

  virtual bool canCompileOffline() const { return true; }

protected:
  poplar::Target target;
};

popart::DeviceType convertDeviceType(poplar::TargetType targetType);
poplar::TargetType convertDeviceType(popart::DeviceType targetType);
void addSyncConfig(const SyncPattern syncPattern, poplar::OptionFlags &flags);

} // namespace popx
} // namespace popart

#endif
