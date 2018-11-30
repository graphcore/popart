#ifndef GUARD_NEURALNET_DEVICEMANAGER_HPP
#define GUARD_NEURALNET_DEVICEMANAGER_HPP

#include <vector>
#include <poponnx/names.hpp>

namespace poponnx {

enum class DeviceType { IpuModel, Cpu, Ipu, Sim };

class DeviceProvider;

// TOD : When the Device/DeviceX classes are renamed to be ?backend? (Need to
// think of a good name) Then we can rename DeviceInfo to Device. THere is no
// seperate Target. is that ok?
/// Represents a device
class DeviceInfo {

public:
  DeviceInfo(DeviceProvider &_provider, DeviceType _type)
      : provider(_provider), type(_type) {
    (void)provider;
  }

  virtual ~DeviceInfo() {}

  /// Attach to the IPU.
  /// \return Returns true if successfully attaches to the device
  virtual bool attach() = 0;

  /// Detach from the IPU.
  virtual void detach() = 0;

  virtual void createVirtualGraph(int tilesPerIpu) = 0;

  /// Get the type of the device
  DeviceType getType() const { return type; }

  /// Return a description of the device
  std::string toString() const;

  /// Get the device id
  virtual int getId() const = 0;
  /// Get the version of the software on the ipu
  virtual std::string getVersion() const = 0;
  /// Get the number of ipus
  virtual int getNumIpus() const = 0;
  /// Get the number tiles per ipu
  virtual int getTilesPerIpu() const = 0;
  /// Get the number work contexts per tile
  virtual int getNumWorkerContexts() const = 0;

  virtual std::vector<unsigned> getDriverIds() const = 0;

private:
  DeviceProvider &provider;
  DeviceType type;
};

std::ostream &operator<<(std::ostream &os, const DeviceInfo &di);

/// The interface for device provides which are registered with the device
/// manager
class DeviceProvider {
public:
  virtual ~DeviceProvider() {}

  /// Get a list of ipu devices
  virtual void enumerate(std::vector<std::unique_ptr<DeviceInfo>> &devices) = 0;

  /// Create a host device for testing
  virtual std::unique_ptr<DeviceInfo>
  createHostDevice(DeviceType type,
                   const std::map<std::string, std::string> &options) = 0;
};

/// The class for
///
class DeviceManager {

  std::vector<DeviceProvider *> providers;

public:
  /** Accessor for the device manager
   * \return A reference to the DeviceManager
   */
  static DeviceManager &getDeviceManager();

  /** Used to register a device provider
   * \param provider A provider
   */
  void registerDeviceProvider(DeviceProvider *provider);

  /** List all the connected hardware devices. To use a device you must first
   * attach the device before you can set it on the Session
   * \return A list of devices.
   */
  std::vector<std::unique_ptr<DeviceInfo>> enumerateDevices();

  /** Finds the first avalible hardware device. This method will attach to the
   * device.
   * \return A device, which can be used with a session. Will return nullptr if
   *         no device is avaliable
   */
  std::unique_ptr<DeviceInfo> acquireAvaliableDevice();

  /** Finds the first avalible hardware device, that a certain number of IPUs.
   * This method will attach to the device.
   * \param numIpus The number of IPUs on the device
   * \param tilesPerIpu The number of tiles on the IPU
   * \return A device, which can be used with a session. Will return nullptr if
   *         no device is avaliable
   */
  std::unique_ptr<DeviceInfo> acquireAvaliableDevice(int numIpus,
                                                     int tilesPerIpu);

  /** Allocates the hardware device by id. This id can be found running 'gc-info
   *  -l' This method will attache to the device
   * \param id The index of the IPU to be used
   * \return A device. Will return nullptr if the device is not  avaliable
   */
  std::unique_ptr<DeviceInfo> acquireDeviceById(int id);

  /** Create a 'simulated' CPU device
   * \return A device
   */
  std::unique_ptr<DeviceInfo> createCpuDevice();

  /** Create a 'simulated' IPU Model device
   * The following options are supported :
   * | numIPUs     | The number of ipus to simulate |
   * | tilesPerIPU | The number of tiles per ipu |
   * | compileIPUCode | Whether or note to compile read Ipu code for modelling|
   * \param options Configuration settings for the IPU Model
   * \return A device
   */
  std::unique_ptr<DeviceInfo>
  createIpuModelDevice(std::map<std::string, std::string> &options);

  /** Create a 'simulated' Sim device
   * The following options are supported :
   * | numIPUs     | The number of ipus to simulate |
   * | tilesPerIPU | The number of tiles per ipu |
   * \param options Configuration settings for the Sim
   * \return A device
   */
  std::unique_ptr<DeviceInfo>
  createSimDevice(std::map<std::string, std::string> &options);
};

/** Write a representation of a DeviceType to an output stream
 *
 * \param os output stream
 * \param dt device type reference
 * \return the same output stream for chaining
 */
std::ostream &operator<<(std::ostream &os, const DeviceType &dt);

} // namespace poponnx

#endif
