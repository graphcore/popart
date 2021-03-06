// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Session_createFromIr
#include <boost/test/unit_test.hpp>

#include <popart/devicemanager.hpp>
#include <popart/session.hpp>

#include <string>

using namespace popart;

namespace {

bool hasPrefix(const std::string &str, const std::string &prefix) {
  return str.length() >= prefix.length() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

std::function<bool(const error &)>
checkErrorMsgFunc(const std::string &expectedPrefix) {

  return [&](const error &ex) -> bool {
    return hasPrefix(ex.what(), expectedPrefix);
  };
}

template <typename SessionTy>
void requireThrowsOnNullDeviceInfo(const std::string &sesionTyName) {
  auto ir = std::make_unique<Ir>();
  const std::shared_ptr<DeviceInfo> di;

  BOOST_REQUIRE_EXCEPTION(
      SessionTy::createFromIr(std::move(ir), di),
      error,
      // Thrown from sesionTyName::createFromIr code.
      checkErrorMsgFunc(sesionTyName +
                        "::createFromIr: Must pass valid DeviceInfo."));
}

template <typename SessionTy>
void requireThrowsOnUnpreparedIr(const std::string &sesionTyName) {
  // TODO(T36404 follow-up): Create Ir whose isPrepared() always returns false,
  // so test is not tied to semantics of default ctor.
  auto ir       = std::make_unique<Ir>();
  const auto di = DeviceManager::createDeviceManager().createCpuDevice();

  BOOST_REQUIRE_EXCEPTION(
      SessionTy::createFromIr(std::move(ir), di),
      error,
      // Thrown from sesionTyName::createFromIr code.
      checkErrorMsgFunc(sesionTyName + "::createFromIr: Ir must be prepared"));
}

} // namespace

BOOST_AUTO_TEST_CASE(TestThrowsOnNullDeviceInfo) {
  requireThrowsOnNullDeviceInfo<InferenceSession>("InferenceSession");
  requireThrowsOnNullDeviceInfo<TrainingSession>("TrainingSession");
}

BOOST_AUTO_TEST_CASE(TestThrowsOnUnpreparedIr) {
  requireThrowsOnUnpreparedIr<InferenceSession>("InferenceSession");
  requireThrowsOnUnpreparedIr<TrainingSession>("TrainingSession");
}
