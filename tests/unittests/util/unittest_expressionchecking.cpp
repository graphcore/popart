// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_expressionchecking

#include "boost/test/auto_unit_test.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/util/expressionchecking.hpp"

std::string makeErrorMessage(const unsigned line, const std::string &message) {
#ifndef NDEBUG
  return popart::logging::format("{}:{} {}", __FILE__, line, message);
#else
  return popart::logging::format("{}", message);
#endif // NDEBUG
}

BOOST_AUTO_TEST_CASE(test_popart_check_fails) {
  std::string expected_message;
  const unsigned zero     = 0;
  const unsigned one      = 1;
  const unsigned *pointer = &one;

  auto errorMessageMatches = [&expected_message](popart::error const &error) {
    return std::string(error.what()) == expected_message;
  };

  expected_message =
      makeErrorMessage(__LINE__ + 2, "Check zero == one has failed. [0 != 1]");
  BOOST_CHECK_EXCEPTION(
      POPART_CHECK_EQ(zero, one), popart::error, errorMessageMatches);

  expected_message =
      makeErrorMessage(__LINE__ + 2, "Check zero >= one has failed. [0 < 1]");
  BOOST_CHECK_EXCEPTION(
      POPART_CHECK_GE(zero, one), popart::error, errorMessageMatches);

  expected_message =
      makeErrorMessage(__LINE__ + 2, "Check zero > one has failed. [0 <= 1]");
  BOOST_CHECK_EXCEPTION(
      POPART_CHECK_GT(zero, one), popart::error, errorMessageMatches);

  expected_message =
      makeErrorMessage(__LINE__ + 2, "Check one <= zero has failed. [1 > 0]");
  BOOST_CHECK_EXCEPTION(
      POPART_CHECK_LE(one, zero), popart::error, errorMessageMatches);

  expected_message =
      makeErrorMessage(__LINE__ + 2, "Check one < zero has failed. [1 >= 0]");
  BOOST_CHECK_EXCEPTION(
      POPART_CHECK_LT(one, zero), popart::error, errorMessageMatches);

  expected_message =
      makeErrorMessage(__LINE__ + 2, "Check one != one has failed. [1 == 1]");
  BOOST_CHECK_EXCEPTION(
      POPART_CHECK_NE(one, one), popart::error, errorMessageMatches);

  expected_message =
      makeErrorMessage(__LINE__ + 2, "Check one != one has failed. [1 == 1]");
  BOOST_CHECK_EXCEPTION(
      POPART_CHECK_NE(one, one), popart::error, errorMessageMatches);

  expected_message =
      makeErrorMessage(__LINE__ + 2, "Check pointer == nullptr has failed.");
  BOOST_CHECK_EXCEPTION(
      POPART_CHECK(pointer == nullptr), popart::error, errorMessageMatches);
}

BOOST_AUTO_TEST_CASE(test_popart_check_fails_with_extra_message) {
  std::string expected_message;
  const unsigned zero     = 0;
  const unsigned one      = 1;
  const unsigned *pointer = &one;

  auto errorMessageMatches = [&expected_message](popart::error const &error) {
    return std::string(error.what()) == expected_message;
  };

  expected_message = makeErrorMessage(
      __LINE__ + 3, "Check zero == one has failed. Uh-oh! [0 != 1]");
  BOOST_CHECK_EXCEPTION(POPART_CHECK_EQ(zero, one) << "Uh-oh!",
                        popart::error,
                        errorMessageMatches);

  expected_message = makeErrorMessage(
      __LINE__ + 3, "Check zero >= one has failed. Uh-oh! [0 < 1]");
  BOOST_CHECK_EXCEPTION(POPART_CHECK_GE(zero, one) << "Uh-oh!",
                        popart::error,
                        errorMessageMatches);

  expected_message = makeErrorMessage(
      __LINE__ + 3, "Check zero > one has failed. Uh-oh! [0 <= 1]");
  BOOST_CHECK_EXCEPTION(POPART_CHECK_GT(zero, one) << "Uh-oh!",
                        popart::error,
                        errorMessageMatches);

  expected_message = makeErrorMessage(
      __LINE__ + 3, "Check one <= zero has failed. Uh-oh! [1 > 0]");
  BOOST_CHECK_EXCEPTION(POPART_CHECK_LE(one, zero) << "Uh-oh!",
                        popart::error,
                        errorMessageMatches);

  expected_message = makeErrorMessage(
      __LINE__ + 3, "Check one < zero has failed. Uh-oh! [1 >= 0]");
  BOOST_CHECK_EXCEPTION(POPART_CHECK_LT(one, zero) << "Uh-oh!",
                        popart::error,
                        errorMessageMatches);

  expected_message = makeErrorMessage(
      __LINE__ + 3, "Check one != one has failed. Uh-oh! [1 == 1]");
  BOOST_CHECK_EXCEPTION(POPART_CHECK_NE(one, one) << "Uh-oh!",
                        popart::error,
                        errorMessageMatches);

  expected_message = makeErrorMessage(
      __LINE__ + 3, "Check pointer == nullptr has failed. Uh-oh!");
  BOOST_CHECK_EXCEPTION(POPART_CHECK(pointer == nullptr) << "Uh-oh!",
                        popart::error,
                        errorMessageMatches);
}

BOOST_AUTO_TEST_CASE(test_popart_check_succeeds) {
  const unsigned *pointer = nullptr;

  BOOST_CHECK_NO_THROW(POPART_CHECK_EQ(1, 1));

  BOOST_CHECK_NO_THROW(POPART_CHECK_GE(1, 0));
  BOOST_CHECK_NO_THROW(POPART_CHECK_GE(1, 1));

  BOOST_CHECK_NO_THROW(POPART_CHECK_GT(1, 0));

  BOOST_CHECK_NO_THROW(POPART_CHECK_LE(0, 1));
  BOOST_CHECK_NO_THROW(POPART_CHECK_LE(1, 1));

  BOOST_CHECK_NO_THROW(POPART_CHECK_LT(0, 1));

  BOOST_CHECK_NO_THROW(POPART_CHECK_NE(0, 1));

  BOOST_CHECK_NO_THROW(POPART_CHECK(pointer == nullptr));
}
