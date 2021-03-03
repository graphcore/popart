// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DEVICEMANAGER_HPP
#define GUARD_NEURALNET_DEVICEMANAGER_HPP

#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <popart/names.hpp>

namespace poplar {
class OptionFlags;
class Target;
} // namespace poplar

namespace popart {

enum class DeviceType { IpuModel = 0, Cpu, Ipu, OfflineIpu, Sim };

enum class SyncPattern { Full = 0, SinglePipeline, ReplicaAndLadder };

enum class DeviceConnectionType { Always = 0, OnDemand, Never };

enum class DeviceSelectionCriterion {
  First = 0, // Select the first device available
  Random     // Select a device randomly from those available
};

SyncPattern syncPatternFromString(const std::string &str);
std::string syncPatternToString(const SyncPattern &pattern);

class DeviceProvider;

// TODO : When the Device/DeviceX classes are renamed to be ?backend? (Need to
// think of a good name) Then we can rename DeviceInfo to Device. There is no
// separate Target. is that ok?
/// Represents a device
class DeviceInfo {

public:
  DeviceInfo(DeviceProvider &_provider,
             DeviceType _type,
             DeviceConnectionType _connectionType,
             const poplar::OptionFlags &_flags);

  virtual ~DeviceInfo();

  /// Attach to the device.
  /// \return True if successfully attached to the device.
  virtual bool attach() = 0;

  /// Detach from the device.
  virtual void detach() = 0;

  /// True if attached.
  virtual bool isAttached() const = 0;

  /// Get the type of the device.
  DeviceType getType() const { return type; }

  /// Get the connection type of the device.
  DeviceConnectionType getConnectionType() const { return connectionType; }

  /// Return a description of the device.
  std::string toString() const;

  /// Get the device id.
  virtual int getId() const = 0;
  /// Get the version of the software on the IPU.
  virtual std::string getVersion() const = 0;
  /// Get the number of IPUs in the device.
  virtual int getNumIpus() const = 0;
  /// Get the number of tiles per IPU.
  virtual int getTilesPerIPU() const = 0;
  /// Get the number of worker contexts per tile.
  virtual int getNumWorkerContexts() const = 0;

  virtual std::vector<unsigned> getDriverIds() const = 0;

  virtual const poplar::Target &getTarget() const = 0;

  // Whether the device supports offlne compilation
  virtual bool canCompileOffline() const { return false; }

  const poplar::OptionFlags &getOptionFlags() const;

  void setOnDemandAttachTimeout(const unsigned seconds);
  const unsigned &getOnDemandAttachTimeout() const { return attachTimeout; }

  bool tryAttachUntilTimeout();

private:
  DeviceProvider &provider;
  DeviceType type;
  DeviceConnectionType connectionType;
  const std::unique_ptr<const poplar::OptionFlags> flags;
  // How many seconds to wait when trying to attach to an IPU
  unsigned attachTimeout = 0;
};

std::ostream &operator<<(std::ostream &os, const DeviceInfo &di);

/// The interface for device providers which are registered with the device
/// manager.
class DeviceProvider {
public:
  virtual ~DeviceProvider() {}

  virtual std::shared_ptr<DeviceInfo>
  getDevice(SyncPattern syncPattern,
            unsigned deviceManagerId,
            DeviceConnectionType connectionType) = 0;

  /**
   * Get the list of all devices fulfilling the specified criteria.
   *
   * \param devices Devices to get.
   * \param requiredNumIPUs Number of IPUs to request.
   * \param syncPattern Sync pattern.
   * \param requiredTilesPerIPU Number of tiles per IPU to request.
   */
  virtual void enumerate(std::vector<std::shared_ptr<DeviceInfo>> &devices,
                         uint32_t requiredNumIPUs,
                         SyncPattern syncPattern,
                         DeviceType type,
                         DeviceConnectionType connectionType,
                         uint32_t requiredTilesPerIPU) = 0;

  /// Create a host device for testing.
  virtual std::shared_ptr<DeviceInfo>
  createHostDevice(DeviceType type,
                   const std::map<std::string, std::string> &options,
                   SyncPattern syncPattern = SyncPattern::Full) = 0;
};

/// A class to manage devices.
///
class DeviceManager {

  std::vector<DeviceProvider *> providers;

public:
  /** Accessor for the device manager.
   * \return A reference to the DeviceManager.
   */
  static DeviceManager &createDeviceManager();

  /** Used to register a device provider.
   * \param provider A provider.
   */
  void registerDeviceProvider(DeviceProvider *provider);

  /**
   * Get the Device object of a device by ID.
   *
   * \param syncPattern Sync pattern.
   * \param deviceManagerId Number of IPUs to request.
   * \return List of requested IPUs.
   */
  std::shared_ptr<DeviceInfo>
  getDevice(SyncPattern syncPattern             = SyncPattern::Full,
            uint32_t deviceManagerId            = 0,
            DeviceConnectionType connectionType = DeviceConnectionType::Always);

  /**
   * Get the list of all devices fulfilling the specified criteria.
   *
   * \param pattern Sync pattern.
   * \param numIpus Number of IPUs to request.
   * \param deviceType Type of device required.
   * \param tilesPerIPU The number of tiles per IPU required.
   * \return List of requested IPUs.
   */
  std::vector<std::shared_ptr<DeviceInfo>> enumerateDevices(
      SyncPattern pattern                 = SyncPattern::Full,
      int numIpus                         = 1,
      DeviceType deviceType               = DeviceType::Ipu,
      DeviceConnectionType connectionType = DeviceConnectionType::Always,
      int tilesPerIPU                     = 0);

  /** Finds the first available hardware device, with a certain number of IPUs.
   * This method will attach to the device.
   * \param numIpus The number of IPUs on the device [=1].
   * \param tilesPerIPU The number of tiles per IPU (0 will match any number)
   * [=0]
   * \return A device, which can be used with a session. Will return
   * nullptr if no device is available.
   */
  std::shared_ptr<DeviceInfo> acquireAvailableDevice(
      int numIpus                         = 1,
      int tilesPerIPU                     = 0,
      SyncPattern pattern                 = SyncPattern::Full,
      DeviceConnectionType connectionType = DeviceConnectionType::Always,
      DeviceSelectionCriterion selectionCriterion =
          DeviceSelectionCriterion::First);

  /** Allocates the hardware device by id. This id can be found running `gc-info
   *  -l`. This method will attach to the device.
   * \param id The index of the IPU to be used.
   * \return A device. Will return nullptr if the device is not available.
   */
  std::shared_ptr<DeviceInfo> acquireDeviceById(
      int id,
      SyncPattern pattern                 = SyncPattern::Full,
      DeviceConnectionType connectionType = DeviceConnectionType::Always);

  /** Create a 'simulated' CPU device.
   * \return A device.
   */
  std::shared_ptr<DeviceInfo> createCpuDevice();

  /** Create a 'simulated' IPU Model device.
   * The following options are supported:
   *
   * * ``numIPUs``:         The number of IPUs to simulate [=1]
   * * ``ge``:     The number of tiles per IPU [=defaultFewTiles]
   * * ``compileIPUCode``:  Whether or not to compile real IPU code for
   *   modelling
   *
   * \param options Configuration settings for the IPU Model.
   * \return A device.
   */
  std::shared_ptr<DeviceInfo>
  createIpuModelDevice(std::map<std::string, std::string> &options);

  /* [INTERNAL]
   * Create a 'simulated' Sim device.
   * The following options are supported:
   *
   * * ``numIPUs``:         The number of IPUs to simulate
   * * ``ge``:     The number of tiles per IPU [=defaultFewTiles]
   * * ``ipuVersion``:     The ipu architecture [="ipu2"]
   *
   * \param options Configuration settings for the Sim.
   * \return A device.
   */
  std::shared_ptr<DeviceInfo>
  createSimDevice(std::map<std::string, std::string> &options);

  /** Create a device resembling an IPU for offline compilation,
   * The following options are supported:
   *
   * * ``numIPUs``:        The number of IPUs to compile for
   * * ``ge``:    The number of tiles per IPU [=defaultManyTiles]
   * * ``ipuVersion``:     The ipu architecture [="ipu1"]
   * * ``syncPattern``:    The sync pattern to use:
   *                       full/singlePipline/replicaAndLadder,
   *                       defaults to full
   *
   * \param options Configuration settings for the IPU Model.
   * \return A device.
   */
  std::shared_ptr<DeviceInfo>
  createOfflineIPUDevice(std::map<std::string, std::string> &options);

  /** If unable to attach to a device on first try, the attach timeout
   * set here is the length of time (in seconds) that the DeviceManager
   * will wait to try and attach. Note: this only takes effect when trying
   * to attach with a DeviceConnectionType::OnDemand DeviceConnectionType.
   * \param seconds The attach timeout in seconds.
   */
  void setOnDemandAttachTimeout(const unsigned seconds);

private:
  // How many seconds to wait when trying to attach to an IPU.
  unsigned attachTimeout = 0;
};

/** Write a representation of a DeviceType to an output stream.
 *
 * \param os Output stream.
 * \param dt Device type reference.
 * \return The same output stream for chaining.
 */
std::ostream &operator<<(std::ostream &os, const DeviceType &dt);

/** Write a representation of a DeviceConnectionType to an output stream.
 *
 * \param os Output stream.
 * \param dct Device connection type reference.
 * \return The same output stream for chaining.
 */
std::ostream &operator<<(std::ostream &os, const DeviceConnectionType &dct);

/** Write a representation of a SyncPattern to an output stream.
 *
 * \param os Output stream.
 * \param sp Sync pattern reference.
 * \return The same output stream for chaining.
 */
std::ostream &operator<<(std::ostream &, const SyncPattern &);

} // namespace popart

namespace std {
template <> struct hash<popart::DeviceInfo> {
  std::size_t operator()(const popart::DeviceInfo &di) const;
};
} // namespace std

#endif
