// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTWRAPPER_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTWRAPPER_HPP_

#include "testfailuretriggerer.hpp" // IWYU pragma: keep
#include <functional>
#include <memory>

namespace popart {

// Forward declaration.
class Ir;

namespace irquery {

/**
 * Base class for TestWrapper objects -- objects that wrap around IR objects to
 * provide easy-to-use test functions. The object we're wrapping must be
 * copyable. Wrap the object in a std::reference_wrapper if need be.
 *
 * Style-guide for TestWrapper functions:
 *
 *  - Functions that get related TestWrappers should having naming like
 *    `inputs`, `ops`.
 *  - Function that perform a boolean test should have a name like `hasOps`
 *    or `hasId` and, if it is conditional on a certain object existing for
 *    which a test wrapper exist it should return a nonstd::optional<...> with
 *    that test wrapper. Also, it should take a `Require` parameter to allow
 *    for negative testing.
 **/
template <typename T> class TestWrapper {
public:
  // Get copy of object we're wrapping.
  T unwrap();
  T unwrap() const;

protected:
  // Constructor.
  TestWrapper(Ir &ir, T wrappedObj);

  // Reference to ir.
  std::reference_wrapper<Ir> ir;
  // Copy of what we're wrapping.
  T wrappedObj;
  // Function IrQuery objects calls to trigger a test failure.
  std::unique_ptr<TestFailureTriggerer> triggerer;
};

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTWRAPPER_HPP_
