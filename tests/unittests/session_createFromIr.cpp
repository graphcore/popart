// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Session_createFromIr
#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/devicemanager.hpp>
#include <popart/session.hpp>
#include <popart/util.hpp>

#include <string>

using namespace popart;

namespace {

std::function<bool(const error &)>
checkErrorMsgFunc(const std::string &expectedPrefix) {

  return [&](const error &ex) -> bool {
    return boost::algorithm::starts_with(ex.what(), expectedPrefix);
  };
}

template <typename SessionTy>
void requireThrowsOnNullDeviceInfo(const std::string &sesionTyName) {
  auto ir = std::shared_ptr<Ir>();
  const std::shared_ptr<DeviceInfo> di;

  BOOST_REQUIRE_EXCEPTION(
      SessionTy::createFromIr(std::move(ir), di),
      error,
      // Thrown from sesionTyName::createFromIr code.
      checkErrorMsgFunc(sesionTyName +
                        "::createFromIr: Must pass valid DeviceInfo."));
}

} // namespace

BOOST_AUTO_TEST_CASE(TestThrowsOnNullDeviceInfo) {
  requireThrowsOnNullDeviceInfo<InferenceSession>("InferenceSession");
  requireThrowsOnNullDeviceInfo<TrainingSession>("TrainingSession");
}
