// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_TESTDEVICE_INCLUDE_TESTDEVICE_HPP_
#define POPART_TESTS_TESTUTIL_TESTDEVICE_INCLUDE_TESTDEVICE_HPP_

#ifdef TEST_WITH_TARGET
/* When TEST_WITH_TARGET is defined this file will cause a main() to be
 * generated that will set TEST_TARGET based on the commandline. The requires
 * that BOOST_TEST_MODULE is defined before including this file, and
 * <boost/test/unit_test.hpp> must NOT be included beforehand.
 */
#ifndef BOOST_TEST_MODULE
#error                                                                         \
    "When TEST_WITH_TARGET is defined BOOST_TEST_MODULE must be defined before including TestDevice.hpp or any boost test headers"
#endif
#endif // TEST_WITH_TARGET

#if defined(TEST_WITH_TARGET) || defined(TEST_WITHOUT_TARGET)
#include <boost/test/unit_test.hpp>
#endif

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <poplar/OptionFlags.hpp>
#include <popart/defaulttilecount.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <boost/asio.hpp>
#include <boost/assign.hpp>
#include <boost/endian/buffers.hpp>
#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include "popart/devicemanager.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"

namespace popart {

// In CMakeLists.txt there is a regex on "Hw*" so be
// careful when adding new enums that begin with Hw:
enum class TestDeviceType {
  Cpu,
  Sim2,
  Sim21,
  Hw,
  IpuModel2,
  IpuModel21,
  OfflineIpu
};

constexpr bool isSimulator(TestDeviceType d) {
  return d == TestDeviceType::Sim2 || d == TestDeviceType::Sim21;
}
constexpr bool isIpuModel(TestDeviceType d) {
  return d == TestDeviceType::IpuModel2 || d == TestDeviceType::IpuModel21;
}
constexpr bool isHw(TestDeviceType d) { return d == TestDeviceType::Hw; }

const char *TestDeviceTypeToIPUName(TestDeviceType d);

const char *asString(const TestDeviceType &deviceType);

TestDeviceType getTestDeviceType(const std::string &deviceString);

std::istream &operator>>(std::istream &is, TestDeviceType &type);

std::ostream &operator<<(std::ostream &os, const TestDeviceType &type);

std::shared_ptr<popart::DeviceInfo> createTestDevice(
    const TestDeviceType testDeviceType,
    const unsigned numIPUs              = 1,
    unsigned tilesPerIPU                = 0,
    const SyncPattern pattern           = SyncPattern::Full,
    const poplar::OptionFlags &opts     = {},
    DeviceConnectionType connectionType = DeviceConnectionType::OnDemand);

} // namespace popart

// TEST_TARGET is defined by TestDevice.cpp. When TEST_WITH_TARGET is defined
// it is initialised by the boost test infrastructure. When TEST_WITH_TARGET is
// not defined explicit initialisation is required.
extern popart::TestDeviceType TEST_TARGET;

#ifdef TEST_WITH_TARGET
// Defines to allow the test device to be specified on the command line, and
// for test predication.

struct CommandLineDeviceInit {
  CommandLineDeviceInit() {
    // configure TEST_TARGET based on the test device argument
    if (boost::unit_test::framework::master_test_suite().argc != 3 ||
        boost::unit_test::framework::master_test_suite().argv[1] !=
            std::string("--device-type"))
      BOOST_FAIL("This test requires the device to be specified on the "
                 "command-line via <test-command> [ctest arguments] -- "
                 "--device-type <device-type>");
    auto deviceString =
        boost::unit_test::framework::master_test_suite().argv[2];
    TEST_TARGET = popart::getTestDeviceType(deviceString);
  }
  void setup() {}
  void teardown() {}
};

// Note this defines main(); BOOST_TEST_MODULE must be defined at this point
BOOST_TEST_GLOBAL_FIXTURE(CommandLineDeviceInit);

struct enableIfSimulator {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(isSimulator(TEST_TARGET));
    ans.message() << "test only supported for simulator targets";
    return ans;
  }
};

struct enableIfHw {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(isHw(TEST_TARGET));
    ans.message() << "test only supported for hardware targets";
    return ans;
  }
};

struct enableIfCpu {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(TEST_TARGET ==
                                            popart::TestDeviceType::Cpu);
    ans.message() << "test only supported for Cpu target";
    return ans;
  }
};

struct enableIfIpuModel {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(isIpuModel(TEST_TARGET));
    ans.message() << "test only supported for IpuModel targets";
    return ans;
  }
};

struct enableIfSimOrHw {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    bool is_sim_target = isSimulator(TEST_TARGET);
    bool is_hw_target  = isHw(TEST_TARGET);
    bool is_ipu_target = is_sim_target || is_hw_target;
    boost::test_tools::assertion_result ans(is_ipu_target);
    ans.message() << "test only supported for Sim and Hw targets";
    return ans;
  }
};

using enableIfInstrumentable = enableIfSimOrHw;

struct enableIfNotCpu {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(TEST_TARGET !=
                                            popart::TestDeviceType::Cpu);
    ans.message() << "test not supported for Cpu target";
    return ans;
  }
};

struct enableIfNotSim {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(!isSimulator(TEST_TARGET));
    ans.message() << "test not supported for simulator targets";
    return ans;
  }
};

struct enableIfIpuModelOrHw {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(isIpuModel(TEST_TARGET) ||
                                            isHw(TEST_TARGET));
    ans.message() << "test only supported for IpuModel and Hw targets";
    return ans;
  }
};

/* Test disabled
 * Used in tests that we don't want to run for now, but we still want them to
 * get built.
 */
struct testDisabled {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(false);
    ans.message() << "test disabled";
    return ans;
  }
};

#endif // TEST_WITH_TARGET

#endif // POPART_TESTS_TESTUTIL_TESTDEVICE_INCLUDE_TESTDEVICE_HPP_
