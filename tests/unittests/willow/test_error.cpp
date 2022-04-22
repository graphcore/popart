// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE WillowErrorUnittest

#include <boost/test/unit_test.hpp>
#include <functional>
#include <string>
#include <popart/error.hpp>

#include "popart/erroruid.hpp"
#include "popart/logging.hpp"

using namespace popart;

template <typename Ex>
std::function<bool(const Ex &)> checkErrorMsgFn(const std::string &expected) {
  return [=](const Ex &ex) -> bool { return ex.what() == expected; };
}

BOOST_AUTO_TEST_CASE(with_error_uid) {
  auto exception = [](auto s) { throw popart::error(ErrorUid::E0, s, 42); };
  const auto checkErrorFn = checkErrorMsgFn<error>("POPART10000: Error 42");

  {
    const std::string s = "Error {}";
    BOOST_REQUIRE_EXCEPTION(exception(s), error, checkErrorFn);
  }

  {
    const char *s = "Error {}";
    BOOST_REQUIRE_EXCEPTION(exception(s), error, checkErrorFn);
  }
}

BOOST_AUTO_TEST_CASE(without_error_uid) {
  auto exception          = [](auto s) { throw popart::error(s, 42); };
  const auto checkErrorFn = checkErrorMsgFn<error>("Error 42");

  {
    const std::string s = "Error {}";
    BOOST_REQUIRE_EXCEPTION(exception(s), error, checkErrorFn);
  }

  {
    const char *s = "Error {}";
    BOOST_REQUIRE_EXCEPTION(exception(s), error, checkErrorFn);
  }
}
