// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_TEST_WRAPPER_HPP
#define TEST_UTILS_IR_QUERY_TEST_WRAPPER_HPP

#include <memory>
#include <testutil/irquery/testfailuretriggerer.hpp>

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

#include <testutil/irquery/testwrapper_impl.hpp>

#endif
