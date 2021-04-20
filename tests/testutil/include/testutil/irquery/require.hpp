// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_REQUIRE_HPP
#define TEST_UTILS_IR_QUERY_REQUIRE_HPP

namespace popart {
namespace irquery {

/**
 * All query functions in this library take a testReq parameter. Leaving it
 * set to `Require::Nothing` will result a function that itself will never
 * trigger a test failure, simply returning a result.
 * ```
 * auto res = foo(Require::Nothing);
 * ```
 *
 * Calling `foo(Require::MustBeTrue)` is used for positive testing and is
 * functionally equivalent to (but has clearer failure messages than):
 * ```
 * auto res = foo(Require::Nothing);  // same as foo(Require::MustBeTrue);
 * BOOST_REQUIRE(res);
 * ```
 *
 * Calling `foo(Require::MustBeFalse)` is used for negative testing and is
 * functionally equivalent to (but has clearer failure messages than):
 * ```
 * auto res = foo(Require::Nothing); // same as foo(Require::MustBeFalse);
 * BOOST_REQUIRE(!res);
 * ```
 *
 * The reason to favour use of testReq arguments over explicit `BOOST_REQUIRE`
 * calls is that you get a more user-friendly failure message.
 **/
enum class Require {
  // If the query result does not hold it constitutes a test failure.
  MustBeTrue = 0,
  // If the query result holds it constitutes a test failure.
  MustBeFalse = 1,
  // The test doesn't fail based on query result.
  Nothing = 2
};

} // namespace irquery
} // namespace popart

#endif
