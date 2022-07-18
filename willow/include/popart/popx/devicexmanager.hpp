// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPDEVICEXMANAGER_HPP
#define GUARD_NEURALNET_POPDEVICEXMANAGER_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <poplar/Device.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Target.hpp>
#include <poplar/TargetType.hpp>
#include <popart/devicemanager.hpp>
#include <popart/error.hpp>

#include "popart/logging.hpp"

namespace popart {
namespace popx {
class Devicex;

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
            DeviceConnectionType connectionType,
            uint32_t requiredTilesPerIPU) override;

  virtual std::shared_ptr<popart::DeviceInfo>
  createHostDevice(popart::DeviceType type,
                   const std::map<std::string, std::string> &options,
                   SyncPattern syncPattern = SyncPattern::Full) override;

  virtual std::shared_ptr<DeviceInfo>
  createOfflineIpuFromDeviceInfo(const DeviceInfo &deviceInfo) override;

  virtual std::shared_ptr<DeviceInfo>
  createOfflineIpuFromSystemString(const std::string &system,
                                   uint32_t numIpus) override;

private:
};

class DevicexInfo : public popart::DeviceInfo {
public:
  DevicexInfo(DeviceProvider &_provider,
              popart::DeviceType _type,
              popart::DeviceConnectionType _connectionType,
              poplar::Device &_device,
              const poplar::OptionFlags &_flags)
      : popart::DeviceInfo(_provider, _type, _connectionType, _flags),
        device(std::move(_device)), isAttached_(false),
        mostRecentlyLoaded(nullptr) {}

  virtual ~DevicexInfo();

  bool attach() override;
  void detach() override;

  int getNumIpus() const override { return getTarget().getNumIPUs(); }
  int getTilesPerIPU() const override { return getTarget().getTilesPerIPU(); }
  int getNumWorkerContexts() const override {
    return getTarget().getNumWorkerContexts();
  }

  std::vector<unsigned> getDriverIds() const override {
    return device.getDriverIDs();
  }

  poplar::Device &getDevice() { return device; }

  const poplar::Target &getTarget() const override {
    return device.getTarget();
  }

  std::string getIpuVersion() const override {
    return getTarget().getTargetArchString();
  }

  virtual bool isAttached() const override { return isAttached_; }

  /**
   * Mark devicex as the last one that was loaded.
   **/
  virtual void setMostRecentlyLoaded(Devicex *devicex);

  /**
   * Check if Devicex was the last one that was loaded.
   **/
  virtual bool isMostRecentlyLoaded(const Devicex *devicex) const;

protected:
  poplar::Device device;
  bool isAttached_;

private:
  // The most recent Devicex that was loaded onto this DevicexInfo's device.
  // This is set for some device when loadEngineAndConnectStreams() is called
  // and is overwritten if another engine is loaded after
  // loadEngineAndConnectStreams() has been called. This is different to
  // 'Devicex::prepareHasBeenCalled_', which, once true, is always true.
  //
  // NOTE: There is no guarantee that this pointer is not dangling. It is
  // possible that the Devicex destructed since this pointer was set. Do not
  // dereference this pointer.
  Devicex *mostRecentlyLoaded;
};

class DevicexCpuInfo : public DevicexInfo {
public:
  DevicexCpuInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider,
                    popart::DeviceType::Cpu,
                    popart::DeviceConnectionType::Always,
                    _device,
                    {}) {}

  virtual int getId() const override { return 0; }
  virtual std::vector<int> getChildIds() const override { return {}; }
  virtual std::string getVersion() const override { return "<unknown-cpu>"; }
};
class DevicexSimInfo : public DevicexInfo {
public:
  DevicexSimInfo(DeviceProvider &_provider, poplar::Device &_device)
      : DevicexInfo(_provider,
                    popart::DeviceType::Sim,
                    popart::DeviceConnectionType::Always,
                    _device,
                    {}) {}

  virtual int getId() const override { return 0; }
  virtual std::vector<int> getChildIds() const override { return {}; }
  virtual std::string getVersion() const override { return "<unknown-sim>"; }
};
class DevicexIpuModelInfo : public DevicexInfo {
public:
  DevicexIpuModelInfo(DeviceProvider &_provider,
                      poplar::Device &_device,
                      const std::string _ipuVersion)
      : DevicexInfo(_provider,
                    popart::DeviceType::IpuModel,
                    popart::DeviceConnectionType::Always,
                    _device,
                    {}),
        ipuVersion(_ipuVersion) {}

  virtual int getId() const override { return 0; }
  virtual std::vector<int> getChildIds() const override { return {}; }
  virtual std::string getVersion() const override { return ipuVersion; }

private:
  std::string ipuVersion;
};
class DevicexIpuInfo : public DevicexInfo {
public:
  DevicexIpuInfo(DeviceProvider &_provider,
                 popart::DeviceConnectionType _dct,
                 int _id,
                 poplar::Device &_device,
                 const poplar::OptionFlags &_flags)
      : DevicexInfo(_provider, popart::DeviceType::Ipu, _dct, _device, _flags),
        id(_id) {}

  virtual int getId() const override { return id; }
  virtual std::vector<int> getChildIds() const override;
  virtual std::string getVersion() const override;

  virtual bool canCompileOffline() const override { return true; }

private:
  int id;
};

class DevicexOfflineIpuInfo : public popart::DeviceInfo {
public:
  DevicexOfflineIpuInfo(DeviceProvider &_provider,
                        poplar::Target &_target,
                        const poplar::OptionFlags &_flags)
      : popart::DeviceInfo(_provider,
                           popart::DeviceType::OfflineIpu,
                           popart::DeviceConnectionType::Never,
                           _flags),
        target(std::move(_target)) {}

  virtual bool attach() override {
    throw error("Cannot attach to offline device");
  }

  virtual void detach() override {
    throw error("Cannot detach from offline device");
  }

  virtual int getId() const override { return 0; }
  virtual std::vector<int> getChildIds() const override { return {}; }
  virtual std::string getVersion() const override { return "<offline-ipu>"; }

  virtual int getNumIpus() const override { return target.getNumIPUs(); }
  virtual int getTilesPerIPU() const override {
    return target.getTilesPerIPU();
  }
  virtual int getNumWorkerContexts() const override {
    return target.getNumWorkerContexts();
  }

  std::string getIpuVersion() const override {
    return getTarget().getTargetArchString();
  }

  virtual std::vector<unsigned> getDriverIds() const override { return {0}; }

  virtual const poplar::Target &getTarget() const override { return target; }

  virtual bool canCompileOffline() const override { return true; }
  virtual bool isAttached() const override { return false; }

protected:
  poplar::Target target;
};

popart::DeviceType convertDeviceType(poplar::TargetType targetType);
poplar::TargetType convertDeviceType(popart::DeviceType targetType);
void addSyncConfig(const SyncPattern syncPattern, poplar::OptionFlags &flags);

} // namespace popx
} // namespace popart

#endif
