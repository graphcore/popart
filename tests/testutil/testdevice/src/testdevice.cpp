// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <testdevice.hpp>

#include <boost/asio.hpp>
#include <boost/assign.hpp>
#include <boost/endian/buffers.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <chrono>
#include <iostream>
#include <thread>

popart::TestDeviceType TEST_TARGET;

namespace popart {

std::shared_ptr<popart::DeviceInfo>
createTestDevice(const TestDeviceType testDeviceType,
                 const unsigned numIPUs,
                 unsigned tilesPerIPU,
                 const SyncPattern pattern,
                 const poplar::OptionFlags &opts,
                 DeviceConnectionType connectionType) {

  if (tilesPerIPU == 0 && ((testDeviceType == TestDeviceType::Sim2) ||
                           (testDeviceType == TestDeviceType::Sim21) ||
                           (testDeviceType == TestDeviceType::IpuModel2) ||
                           (testDeviceType == TestDeviceType::IpuModel21))) {
    // We need a number of tiles for these device types.
    tilesPerIPU = defaultFewTiles;
  }

  std::map<std::string, std::string> deviceOpts{
      {"numIPUs", std::to_string(numIPUs)},
      {"tilesPerIPU", std::to_string(tilesPerIPU)},
  };

  deviceOpts.insert(opts.begin(), opts.end());

  auto &dm = DeviceManager::createDeviceManager();
  switch (testDeviceType) {
  case TestDeviceType::Cpu:
    return dm.createCpuDevice();
  case TestDeviceType::Sim2:
  case TestDeviceType::Sim21:
    deviceOpts.insert({"ipuVersion", TestDeviceTypeToIPUName(testDeviceType)});
    return dm.createSimDevice(deviceOpts);
  case TestDeviceType::OfflineIpu:
    return dm.createOfflineIPUDevice(deviceOpts);
  case TestDeviceType::Hw:
    // Keep trying to attach for 15 minutes before aborting
    dm.setOnDemandAttachTimeout(900);
    return dm.acquireAvailableDevice(numIPUs,
                                     tilesPerIPU,
                                     pattern,
                                     connectionType,
                                     DeviceSelectionCriterion::Random);
  case TestDeviceType::IpuModel2:
  case TestDeviceType::IpuModel21:
    deviceOpts.insert({"ipuVersion", TestDeviceTypeToIPUName(testDeviceType)});
    return dm.createIpuModelDevice(deviceOpts);
  default:
    throw error("Unrecognized device {}", testDeviceType);
  }
}

const char *TestDeviceTypeToIPUName(TestDeviceType d) {
  switch (d) {
  case TestDeviceType::Sim2:
  case TestDeviceType::IpuModel2:
    return "ipu2";

  case TestDeviceType::Sim21:
  case TestDeviceType::IpuModel21:
    return "ipu21";

  case TestDeviceType::Cpu:
    throw popart::error(
        "TestTestDeviceTypeToIPUName(TestTestDeviceType::Cpu) not supported");
  case TestDeviceType::Hw:
    throw popart::error(
        "TestTestDeviceTypeToIPUName(TestTestDeviceType::Hw) not supported");

  default:
    throw popart::error("Unknown device type");
  }
}

std::istream &operator>>(std::istream &is, TestDeviceType &type) {
  std::string token;
  is >> token;
  if (token == "Cpu")
    type = TestDeviceType::Cpu;
  else if (token == "IpuModel2")
    type = TestDeviceType::IpuModel2;
  else if (token == "IpuModel21")
    type = TestDeviceType::IpuModel21;
  else if (token == "Sim2")
    type = TestDeviceType::Sim2;
  else if (token == "Sim21")
    type = TestDeviceType::Sim21;
  else if (token == "Hw")
    type = TestDeviceType::Hw;
  else if (token == "OfflineIpu")
    type = TestDeviceType::OfflineIpu;
  else
    throw std::logic_error(
        "Unsupported device type <" + token +
        ">; must be one of "
        R"XX("Cpu", "IpuModel2", "IpuModel21", "Sim2", "Sim21", or "Hw")XX");
  return is;
}

std::ostream &operator<<(std::ostream &os, const TestDeviceType &type) {
  os << asString(type);
  return os;
}

const char *asString(const TestDeviceType &TestDeviceType) {
  switch (TestDeviceType) {
  case TestDeviceType::Cpu:
    return "Cpu";
  case TestDeviceType::IpuModel2:
    return "IpuModel2";
  case TestDeviceType::IpuModel21:
    return "IpuModel21";
  case TestDeviceType::Sim2:
    return "Sim2";
  case TestDeviceType::Sim21:
    return "Sim21";
  case TestDeviceType::Hw:
    return "Hw";
  case TestDeviceType::OfflineIpu:
    return "OfflineIpu";
  default:
    break;
  }
  throw std::logic_error("Invalid device type");
}

TestDeviceType getTestDeviceType(const std::string &deviceString) {
  if (deviceString == "Cpu")
    return TestDeviceType::Cpu;
  if (deviceString == "IpuModel2")
    return TestDeviceType::IpuModel2;
  if (deviceString == "IpuModel21")
    return TestDeviceType::IpuModel21;
  if (deviceString == "Sim2")
    return TestDeviceType::Sim2;
  if (deviceString == "Sim21")
    return TestDeviceType::Sim21;
  if (deviceString == "Hw")
    return TestDeviceType::Hw;
  if (deviceString == "OfflineIpu")
    return TestDeviceType::OfflineIpu;
  throw std::logic_error("Invalid device string");
}

} // namespace popart
