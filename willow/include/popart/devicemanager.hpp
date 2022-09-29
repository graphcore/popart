// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_DEVICEMANAGER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_DEVICEMANAGER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace poplar {
class OptionFlags;
class Target;
} // namespace poplar

namespace popart {

/// Defines the type of device to use for graph compilation and execution.
enum class DeviceType {

  /**
   * Use the Poplar IPU Model for graph compilation and execution.
   * The IPU Model will simulate the behaviour of the IPU hardware. It will not
   * completely implement every aspect of a real IPU. (Default).
   */
  IpuModel = 0,

  /**
   * Use CPU for graph compilation and execution.
   */
  Cpu,

  /**
   * Use IPU for graph execution.
   */
  Ipu,

  /**
   * Compile graph for later execution.
   * This can be done even if IPUs are not present.
   * Offline graph compilation is also useful for verifying memory constraints.
   */
  OfflineIpu,

  /**
   * [For Graphcore internal use only]
   * Use a simulator for graph compilation and execution.
   */
  Sim
};

/// Controls synchronisation in multi-IPU systems.
enum class SyncPattern {

  /**
   * Require all IPUs to synchronise on every communication between IPUs or
   * between IPUs and host (Default).
   */
  Full = 0,

  /**
   * Allow IPUs to synchronise with the host independently, without having to
   * synchronise with each other. This permits any one IPU to perform host IO
   * while other IPUs are processing data.
   */
  SinglePipeline,

  /**
   * Allow an IPU group to communicate with the host without requiring
   * synchronisation between groups. This permits multiple IPU groups to
   * alternate between performing host IO and computation.
   */
  ReplicaAndLadder

};

/// Controls when to connect to the IPU (if at all).
enum class DeviceConnectionType {

  /**
   * Attach to the IPU from the start (Default).
   */
  Always = 0,

  /**
   * Wait until the compilation is complete and the executable is ready to be
   * run before attaching to the IPU.
   */
  OnDemand,

  /**
   * Never try to attach to an IPU.
   * This is useful for offline compilation (`DeviceType::OfflineIpu`.
   * Trying to run an executable will throw an error.
   */
  Never
};

/// Controls how to select an available IPU.
enum class DeviceSelectionCriterion {
  First = 0, /// Select the first device available. (Default).
  Random     /// Select a device randomly from those available.
};

/**
 * Convert string input to a valid SyncPattern enum value.
 * Throws an error if the input string does not correspond to a valid
 * SyncPattern value.
 * \param str The input string to be converted to a valid SyncPattern enum
 * value. \returns A valid SyncPattern enum value.
 */
SyncPattern syncPatternFromString(const std::string &str);

/**
 * Convert a valid SyncPattern enum value to a string.
 * Throws an error if the input does not correspond to a valid
 * SyncPattern value.
 * \param pattern The input SyncPattern enum value to be converted to a string.
 * \returns The string version of a valid SyncPattern enum value.
 */
std::string syncPatternToString(const SyncPattern &pattern);

class DeviceProvider;

// TODO : When the Device/DeviceX classes are renamed to be ?backend? (Need to
// think of a good name) Then we can rename DeviceInfo to Device. There is no
// separate Target. is that ok?
/// Represents a specific device.
class DeviceInfo {

public:
  /**
   * Constructor for the DeviceInfo class.
   * \param _provider The DeviceProvider instance for the device.
   * \param _type The type of the device.
   * \param _connectionType The setting for when to connect to the device, if at
   *        all.
   * \param _flags A set of Poplar option/value string flags.
   */
  DeviceInfo(DeviceProvider &_provider,
             DeviceType _type,
             DeviceConnectionType _connectionType,
             const poplar::OptionFlags &_flags);

  /// Destructor for DeviceInfo.
  virtual ~DeviceInfo();

  /**
   * Attach to the device.
   *
   * \return `true` if successfully attached to the device, `false` otherwise.
   */
  virtual bool attach() = 0;

  /// Detach from the device.
  virtual void detach() = 0;

  /**
   * Check if attached to the device.
   *
   * \return `true` if attached to the device, `false` otherwise.
   */
  virtual bool isAttached() const = 0;

  /**
   * Get the type of the device.
   *
   * \return The type of the device.
   */
  DeviceType getType() const { return type; }

  /**
   * Get the setting for when to connect to the device.
   *
   * \return The setting for when to connect to the device.
   */
  DeviceConnectionType getConnectionType() const { return connectionType; }

  /// Return a description of the device.
  std::string toString() const;

  /// Get the device id.
  virtual int getId() const = 0;

  /**
   * Get the child device IDs.
   * The value returned by `getId()` for a multi-IPU
   * device is a 'parent ID' and does not relate to the IDs of the devices
   * it comprises. This function, in the case of real devices, uses the Poplar
   * API to work out which single-IPU device IDs it relates to. In the case
   * of replication, a device includes all IPUs involved, so a 2-IPU model with
   * 2x replication would expect to have 4 child IDs returned here.
   */
  virtual std::vector<int> getChildIds() const = 0;

  /// Get the version of the software on the IPU.
  virtual std::string getVersion() const = 0;
  /// Get the number of IPUs in the device.
  virtual int getNumIpus() const = 0;
  /// Get the number of tiles per IPU.
  virtual int getTilesPerIPU() const = 0;
  /// Get the number of worker contexts per tile.
  virtual int getNumWorkerContexts() const = 0;
  /// Get the IPU version.
  virtual std::string getIpuVersion() const = 0;
  /// Get the version of the drivers on the IPU.
  virtual std::vector<unsigned> getDriverIds() const = 0;
  /// Get the Poplar target.
  virtual const poplar::Target &getTarget() const = 0;

  /**
   * Get whether the device supports offline compilation.
   * \returns `true if the device supports offline compilation, otherwise
   *       `false`.
   */
  virtual bool canCompileOffline() const { return false; }

  const poplar::OptionFlags &getOptionFlags() const;

  /**
   * Set timeout (in seconds) for trying to attach to a device.
   * If unable to attach to a device on the first try, the DeviceManager
   * instance will periodically try to attach to the device until successfully
   * attached or this timeout is reached.
   *
   * \note This only applies when trying to attach with
   *      DeviceConnectionType::OnDemand.
   *
   * \param seconds The timeout (in seconds) for trying to attach to the device.
   */
  void setOnDemandAttachTimeout(const unsigned seconds);

  /**
   * Get timeout (in seconds) for trying to attach to a device.
   * \returns The timeout (in seconds) for trying to attach to
   *      the device.
   */
  const unsigned &getOnDemandAttachTimeout() const { return attachTimeout; }

  /**
   * Periodically try to attach to the device until either the attach timeout is
   * reached or successfully attached.
   */
  bool tryAttachUntilTimeout();

  // Return true if this device represents real hardware
  // (i.e it is not a model or a simulator).
  bool isHwCompatible() const;

  /**
   * Log an event for device debugging purposes.
   * This event will get logged to the file location defined by the environment
   * variable POPART_LOG_DEVICE_ACCESS_IN_TESTS, if it is set.
   * \param event A text description of the event to be written to the log.
   * \param auxKeyVals Optional additional parameters to log.
   */
  void writeToDeviceAccessLog(
      const std::string &event,
      const std::map<std::string, std::string> &auxKeyVals = {});

private:
  DeviceProvider &provider;
  DeviceType type;
  DeviceConnectionType connectionType;
  const std::unique_ptr<const poplar::OptionFlags> flags;
  // How many seconds to wait when trying to attach to an IPU
  unsigned attachTimeout = 0;

  // Format of device access log stored so we can log in the destructor.
  std::string deviceAccessLogEntryFmt;
};

std::ostream &operator<<(std::ostream &os, const DeviceInfo &di);

/**
 * The interface for device providers which are registered with the device
 * manager.
 */
class DeviceProvider {
public:
  /// Destructor for DeviceProvider.
  virtual ~DeviceProvider() {}

  /**
   * Get the list of all devices that satisfy the specified criteria.
   * Throws an error if the connection type is DeviceConnectionType::Never.
   *
   * \param syncPattern The setting for synchronisation on multi-IPU systems.
   * \param deviceManagerId The ID of the requested device.
   * \param connectionType The setting for when to connect to the device.
   * \returns The list of all devices that satisfy the specified criteria.
   */
  virtual std::shared_ptr<DeviceInfo>
  getDevice(SyncPattern syncPattern,
            unsigned deviceManagerId,
            DeviceConnectionType connectionType) = 0;

  /**
   * Get the list of all devices that satisfy the specified criteria.
   *
   * \param devices The list of devices.
   * \param requiredNumIPUs The number of IPUs required.
   * \param syncPattern The setting for when to synchronise in a multi-IPU
   *      system.
   * \param type The type of the device to use for compilation and execution.
   * \param connectionType The setting for when to connect to the device.
   * \param requiredTilesPerIPU The number of tiles per IPU required.
   */
  virtual void enumerate(std::vector<std::shared_ptr<DeviceInfo>> &devices,
                         uint32_t requiredNumIPUs,
                         SyncPattern syncPattern,
                         DeviceType type,
                         DeviceConnectionType connectionType,
                         uint32_t requiredTilesPerIPU) = 0;

  /**
   * Create a host device for testing.
   *
   * \param type The type of the device to use for compilation and execution.
   * \param options The configuration for the created device. See
   *      createCpuDevice(), createIpuModelDevice(), createOfflineIPUDevice()
   *      and createSimDevice() for more information about \c options.
   * \param syncPattern The setting for when to synchronise in a multi-IPU
   *      system.
   * \returns The device for use in testing.
   */
  virtual std::shared_ptr<DeviceInfo>
  createHostDevice(DeviceType type,
                   const std::map<std::string, std::string> &options,
                   SyncPattern syncPattern = SyncPattern::Full) = 0;

  virtual std::shared_ptr<DeviceInfo>
  createOfflineIpuFromDeviceInfo(const DeviceInfo &deviceInfo) = 0;

  virtual std::shared_ptr<DeviceInfo>
  createOfflineIpuFromSystemString(const std::string &system,
                                   uint32_t numIpus) = 0;
};

/// A class to manage devices.
class DeviceManager {
private: // data
  std::vector<DeviceProvider *> providers;

  // How many seconds to continue trying to attach to an IPU.
  unsigned attachTimeout = 0;

private: // methods
  // Meyer’s singleton pattern
  DeviceManager()                      = default;
  ~DeviceManager()                     = default;
  DeviceManager(const DeviceManager &) = delete;
  DeviceManager &operator=(const DeviceManager &) = delete;

public: // methods
  /**
   * Accessor for the device manager.
   * \return A reference to the DeviceManager instance.
   */
  static DeviceManager &createDeviceManager();

  /**
   * Register a device provider.
   * \param provider The device provider to be registered with the device
   *       manager.
   */
  void registerDeviceProvider(DeviceProvider *provider);

  /**
   * Get the list of all devices that satisfy the specified criteria.
   *
   * \param devices The list of devices.
   * \param requiredNumIPUs The number of IPUs required.
   * \param syncPattern The setting for when to synchronise in a multi-IPU
   *      system.
   * \param type The type of the device to use for compilation and execution.
   * \param connectionType The setting for when to connect to the device.
   * \param requiredTilesPerIPU The number of tiles per IPU required.
   */
  virtual void
  enumerate(std::vector<std::shared_ptr<popart::DeviceInfo>> &devices,
            unsigned requiredNumIPUs,
            SyncPattern syncPattern,
            DeviceType type,
            DeviceConnectionType connectionType,
            uint32_t requiredTilesPerIPU);

  /**
   * Get the list of all devices with the required criteria.
   *
   * \param pattern The setting for when to synchronise in a multi-IPU
   *      system. (Default: SyncPattern::Full).
   * \param numIpus The number of IPUs required. (Default: 1).
   * \param deviceType The type of the device required. (Default:
   *       DeviceType::Ipu).
   * \param connectionType The setting for when to connect to the device.
   *       (Default: DeviceConnectionType::Always).
   * \param tilesPerIPU The number of tiles per IPU required. (Default: 0).
   * \return The list of devices with the required criteria.
   */
  std::vector<std::shared_ptr<DeviceInfo>> enumerateDevices(
      SyncPattern pattern                 = SyncPattern::Full,
      int numIpus                         = 1,
      DeviceType deviceType               = DeviceType::Ipu,
      DeviceConnectionType connectionType = DeviceConnectionType::Always,
      int tilesPerIPU                     = 0);

  /**
   * Get a device with the required criteria.
   *
   * \param syncPattern The setting for when to synchronise in a multi-IPU
   *      system. (Default: SyncPattern::Full).
   * \param deviceManagerId The ID of the requested device. (Default: 0)
   * \param connectionType The setting for when to connect to the device.
   *      (Default: DeviceConnectionType::Always).
   * \return A device, which can be used with a session. If no device is
   *      acquired, a nullptr is returned.
   */
  std::shared_ptr<DeviceInfo>
  getDevice(SyncPattern syncPattern             = SyncPattern::Full,
            uint32_t deviceManagerId            = 0,
            DeviceConnectionType connectionType = DeviceConnectionType::Always);

  /**
   * Finds an available hardware device, with the specified number of IPUs.
   * This method will attach to the device if \c connectionType is equal to
   * DeviceConnectionType::Always. This method is suitable when polling for an
   * available device when resources are constrained.
   * \param numIpus The number of IPUs on the device (Default: 1).
   * \param tilesPerIPU The number of tiles per IPU. An input of 0 will match
   *      any number. (Default: 0).
   * \param pattern The setting for when to synchronise in a multi-IPU
   *      system. (Default: SyncPattern::Full).
   * \param connectionType The setting for when to connect to the device.
   *       (Default: DeviceConnectionType::Always).
   * \param selectionCriterion The method for selecting a device from the list
   *       of valid selections. (Default: DeviceSelectionCriterion::First).
   * \return A device, which can be used with a session. If no device is
   *       acquired, a nullptr is returned.
   */
  std::shared_ptr<DeviceInfo> tryAcquireAvailableDevice(
      int numIpus                         = 1,
      int tilesPerIPU                     = 0,
      SyncPattern pattern                 = SyncPattern::Full,
      DeviceConnectionType connectionType = DeviceConnectionType::Always,
      DeviceSelectionCriterion selectionCriterion =
          DeviceSelectionCriterion::First);

  /**
   * Finds an available hardware device, with a certain number of IPUs.
   * This method will attach to the device if \c connectionType is equal to
   * DeviceConnectionType::Always.
   * Throws an error if there are less than \c numIpus IPUs available.
   * \param numIpus The number of IPUs on the device [=1].
   * \param tilesPerIPU The number of tiles per IPU. An input of 0 will match
   *      any number. (Default: 0).
   * \param pattern The setting for when to synchronise in a multi-IPU
   *      system. (Default: SyncPattern::Full).
   * \param connectionType The connection type, for deciding when to attach to
   *      the device.
   * \param selectionCriterion How to select a device from the list of valid
   *      selections.
   * \return A device, which can be used with a session.
   */
  std::shared_ptr<DeviceInfo> acquireAvailableDevice(
      int numIpus                         = 1,
      int tilesPerIPU                     = 0,
      SyncPattern pattern                 = SyncPattern::Full,
      DeviceConnectionType connectionType = DeviceConnectionType::Always,
      DeviceSelectionCriterion selectionCriterion =
          DeviceSelectionCriterion::First);

  /**
   * Allocates the hardware device by ID. This ID can be found running `gc-info
   *  -l`. This method will try to attach to the device if \c connectionType is
   *  equal to DeviceConnectionType::Always. This method is suitable when
   * polling for an available device when resources are constrained.
   *
   * \param id The ID of the IPU to be used.
   * \param pattern The setting for when to synchronise in a multi-IPU system.
   *      (Default: SyncPattern::Full).
   * \param connectionType The connection type, for deciding when to attach to
   *      the device. (Default: DeviceConnectionType::Always).
   * \return A device, which can be used with a session. If no device is
   * acquired, a nullptr is returned.
   */
  std::shared_ptr<DeviceInfo> tryAcquireDeviceById(
      int id,
      SyncPattern pattern                 = SyncPattern::Full,
      DeviceConnectionType connectionType = DeviceConnectionType::Always);

  /**
   * Allocates the hardware device by ID. This ID can be found running `gc-info
   *  -l`. This method will attach to the device if \c connectionType is equal
   * to DeviceConnectionType::Always.
   * \param id The ID of the IPU to be used.
   * \param pattern The setting for when to synchronise in a multi-IPU
   *      system. (Default: SyncPattern::Full).
   * \param connectionType The connection type, for deciding when to attach to
   *      the device. (Default: DeviceConnectionType::Always).
   * \return A device, which can be used with a session.
   */
  std::shared_ptr<DeviceInfo> acquireDeviceById(
      int id,
      SyncPattern pattern                 = SyncPattern::Full,
      DeviceConnectionType connectionType = DeviceConnectionType::Always);

  /**
   * Create a simulated device on the host for testing purposes.
   *
   * \param type The type of device to simulate.
   * \param options The configuration settings for the host device.
   * \return The requested device for testing purposes.
   */
  std::shared_ptr<DeviceInfo>
  createHostDevice(DeviceType type,
                   const std::map<std::string, std::string> &options);

  /**
   * Create a simulated CPU device for testing purposes.
   * \return A simulated CPU device.
   */
  std::shared_ptr<DeviceInfo> createCpuDevice();

  /**
   * Create a simulated `IpuModel` device for testing purposes.
   * The following options are supported:
   *
   *    - `numIPUs`: The number of IPUs to simulate (Default: 1).
   *    - `ge`: The number of tiles per IPU (Default: defaultFewTiles).
   *    - `compileIPUCode`:  Indicate whether or not to compile real IPU code
   *           for modelling.
   *
   * \param options Configuration settings for the IPU Model.
   * \return A device.
   */
  std::shared_ptr<DeviceInfo>
  createIpuModelDevice(std::map<std::string, std::string> &options);

  /*
   * [For Graphcore internal use only]
   * Create a simulated `Sim` device for testing purposes.
   * The following options are supported:
   *
   *    - `numIPUs`: The number of IPUs to simulate.
   *    - `ge`: The number of tiles per IPU (Default: defaultFewTiles).
   *    - `ipuVersion`: The ipu architecture (Default: "ipu2").
   *
   * \param options Configuration settings for the Sim.
   * \return A simulated `Sim` device.
   */
  std::shared_ptr<DeviceInfo>
  createSimDevice(std::map<std::string, std::string> &options);

  /**
   * Create a simulated `OfflineIpu` device for testing purposes.
   * This resembles an IPU and is used for offline compilation.
   *
   * The following options are supported:
   *
   *    - `numIPUs`: The number of IPUs to compile for
   *    - `ge`: The number of tiles per IPU (Default: defaultManyTiles).
   *    - `ipuVersion`: The ipu architecture (Default: "ipu1").
   *    - `syncPattern`: The setting for synchronisation in a multi-IPU system.
   *
   * \param options Configuration settings for the IPU Model.
   * \return A simulated `OfflineIpu` device.
   */
  std::shared_ptr<DeviceInfo>
  createOfflineIPUDevice(std::map<std::string, std::string> &options);

  /**
   * Create a simulated `OfflineIpu` device from the description of another
   * device.
   *
   * \param deviceInfo The device to create a `OfflineIpu` version of.
   * \return An `OfflineIpu` device.
   */
  std::shared_ptr<DeviceInfo>
  createOfflineIpuFromDeviceInfo(const DeviceInfo &deviceInfo);

  /**
   * Create a simulated `OfflineIpu` device from the name of a system.
   *
   * \param system The device to create a `OfflineIpu` version of.
   * \param numIpus The number of IPUs. Providing 0 corresponds to all IPUs in
   * system \return An `OfflineIpu` device.
   */
  std::shared_ptr<DeviceInfo>
  createOfflineIpuFromSystemString(const std::string &system, uint32_t numIpus);

  /** If unable to attach to a device on first try, the attach timeout
   * set here is the length of time (in seconds) that the DeviceManager
   * will wait to try and attach. Note: this only takes effect when trying
   * to attach with a DeviceConnectionType::OnDemand DeviceConnectionType.
   * \param seconds The attach timeout in seconds.
   */
  void setOnDemandAttachTimeout(const unsigned seconds);
};

/**
 * Write a representation of a DeviceType object to an output stream.
 *
 * \param os The output stream to write to.
 * \param dt The device type reference to be written to the output
 *        stream.
 * \return The same output stream for chaining.
 */
std::ostream &operator<<(std::ostream &os, const DeviceType &dt);

/**
 * Write a representation of a DeviceConnectionType object to an output stream.
 *
 * \param os The output stream to write to.
 * \param dct The device connection type reference to be written to the output
 *        stream.
 * \return The same output stream for chaining.
 */
std::ostream &operator<<(std::ostream &os, const DeviceConnectionType &dct);

/**
 * Write a representation of a SyncPattern object to an output stream.
 *
 * \param os The output stream to write to.
 * \param sp The sync pattern reference to be written to the output
 *        stream.
 * \return The same output stream for chaining.
 */
std::ostream &operator<<(std::ostream &, const SyncPattern &);

} // namespace popart

namespace std {
template <> struct hash<popart::DeviceInfo> {
  std::size_t operator()(const popart::DeviceInfo &di) const;
};
} // namespace std

#endif // POPART_WILLOW_INCLUDE_POPART_DEVICEMANAGER_HPP_
