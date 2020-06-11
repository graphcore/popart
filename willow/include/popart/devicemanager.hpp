// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DEVICEMANAGER_HPP
#define GUARD_NEURALNET_DEVICEMANAGER_HPP

#include <memory>
#include <sstream>
#include <vector>
#include <poplar/DeviceManager.hpp>
#include <popart/names.hpp>

namespace popart {

enum class DeviceType { IpuModel = 0, Cpu, Ipu, Sim };

enum class SyncPattern { Full = 0, SinglePipeline, PingPong };

enum class DeviceConnectionType { Always = 0, OnDemand, Never };

class DeviceProvider;

// TOD : When the Device/DeviceX classes are renamed to be ?backend? (Need to
// think of a good name) Then we can rename DeviceInfo to Device. THere is no
// seperate Target. is that ok?
/// Represents a device
class DeviceInfo {

public:
  DeviceInfo(
      DeviceProvider &_provider,
      DeviceType _type,
      DeviceConnectionType _connectionType = DeviceConnectionType::Always)
      : provider(_provider), type(_type), connectionType(_connectionType) {
    (void)provider;
  }

  virtual ~DeviceInfo() {}

  /// Attach to the IPU.
  /// \return Returns true if successfully attaches to the device
  virtual bool attach() = 0;

  /// Detach from the IPU.
  virtual void detach() = 0;

  virtual void createVirtualGraph(int tilesPerIpu) = 0;

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
  virtual int getTilesPerIpu() const = 0;
  /// Get the number of worker contexts per tile.
  virtual int getNumWorkerContexts() const = 0;

  virtual std::vector<unsigned> getDriverIds() const = 0;

private:
  DeviceProvider &provider;
  DeviceType type;
  DeviceConnectionType connectionType;
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
   * \param devices Devices to get
   * \param requiredNumIPUs Number of IPUs to request.
   * \param syncPattern Sync pattern
   */
  virtual void enumerate(std::vector<std::shared_ptr<DeviceInfo>> &devices,
                         uint32_t requiredNumIPUs,
                         SyncPattern syncPattern,
                         DeviceType type,
                         DeviceConnectionType connectionType) = 0;

  /// Create a host device for testing
  virtual std::shared_ptr<DeviceInfo>
  createHostDevice(DeviceType type,
                   const std::map<std::string, std::string> &options) = 0;
};

/// A class to manage devices.
///
class DeviceManager {

  std::vector<DeviceProvider *> providers;

public:
  /** Accessor for the device manager.
   * \return A reference to the DeviceManager
   */
  static DeviceManager &createDeviceManager();

  /** Used to register a device provider.
   * \param provider A provider
   */
  void registerDeviceProvider(DeviceProvider *provider);

  /**
   * Get the Device object of a device by ID.
   *
   * \param syncPattern Sync pattern
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
   * \return List of requested IPUs.
   */
  std::vector<std::shared_ptr<DeviceInfo>> enumerateDevices(
      SyncPattern pattern                 = SyncPattern::Full,
      int numIpus                         = 1,
      DeviceType deviceType               = DeviceType::Ipu,
      DeviceConnectionType connectionType = DeviceConnectionType::Always);

  /** Finds the first available hardware device, with a certain number of IPUs.
   * This method will attach to the device.
   * \param numIpus The number of IPUs on the device [=1]
   * \param tilesPerIpu The number of tiles on the IPU (0 will match any number)
   * [=0]
   * \return A device, which can be used with a session. Will return
   * nullptr if no device is available
   */
  std::shared_ptr<DeviceInfo> acquireAvailableDevice(
      int numIpus                         = 1,
      int tilesPerIpu                     = 0,
      SyncPattern pattern                 = SyncPattern::Full,
      DeviceConnectionType connectionType = DeviceConnectionType::Always);

  /** Allocates the hardware device by id. This id can be found running 'gc-info
   *  -l'. This method will attach to the device.
   * \param id The index of the IPU to be used
   * \return A device. Will return nullptr if the device is not available
   */
  std::shared_ptr<DeviceInfo> acquireDeviceById(
      int id,
      SyncPattern pattern                 = SyncPattern::Full,
      DeviceConnectionType connectionType = DeviceConnectionType::Always);

  /** Create a 'simulated' CPU device.
   * \return A device
   */
  std::shared_ptr<DeviceInfo> createCpuDevice();

  /** Create a 'simulated' IPU Model device.
   * The following options are supported:
   *
   * * ``numIPUs``:         The number of IPUs to simulate [=1]
   * * ``tilesPerIPU``:     The number of tiles per IPU [=1216]
   * * ``compileIPUCode``:  Whether or not to compile real IPU code for
   *   modelling
   *
   * \param options Configuration settings for the IPU Model
   * \return A device
   */
  std::shared_ptr<DeviceInfo>
  createIpuModelDevice(std::map<std::string, std::string> &options);

  /* [INTERNAL]
   * Create a 'simulated' Sim device.
   * The following options are supported:
   *
   * * ``numIPUs``:         The number of IPUs to simulate
   * * ``tilesPerIPU``:     The number of tiles per IPU
   *
   * \param options Configuration settings for the Sim
   * \return A device
   */
  std::shared_ptr<DeviceInfo>
  createSimDevice(std::map<std::string, std::string> &options);
};

/** Write a representation of a DeviceType to an output stream.
 *
 * \param os Output stream
 * \param dt Device type reference
 * \return The same output stream for chaining
 */
std::ostream &operator<<(std::ostream &os, const DeviceType &dt);

/** Write a representation of a DeviceConnectionType to an output stream.
 *
 * \param os Output stream
 * \param dct Device connection type reference
 * \return The same output stream for chaining
 */
std::ostream &operator<<(std::ostream &os, const DeviceConnectionType &dct);

} // namespace popart

namespace std {
template <> struct hash<popart::DeviceInfo> {
  std::size_t operator()(const popart::DeviceInfo &di) const {
    // Hash based on all the DeviceManager attributes that
    // can affect compiled program

    std::stringstream ss;
    ss << di.getType() << di.getConnectionType();

    return std::hash<std::string>()(ss.str()) ^
           std::hash<std::string>()(di.getVersion()) ^
           std::hash<int>()(di.getNumIpus()) ^
           std::hash<int>()(di.getTilesPerIpu()) ^
           std::hash<int>()(di.getNumWorkerContexts());
  }
};
}; // namespace std

#endif
