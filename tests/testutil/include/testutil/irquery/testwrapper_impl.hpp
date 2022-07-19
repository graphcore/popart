// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTWRAPPER_IMPL_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTWRAPPER_IMPL_HPP_

#include "testutil/irquery/testwrapper.hpp"
#include <memory>

namespace popart {
class Ir;

namespace irquery {
class TestFailureTriggerer;

// Implementation of TestWrapper::TestWrapper.
template <typename T>
TestWrapper<T>::TestWrapper(Ir &ir_, T wrappedObj_)
    : ir{ir_}, wrappedObj{wrappedObj_},
      triggerer(std::make_unique<TestFailureTriggerer>()) {}

// Implementation of TestWrapper::get.
template <typename T> T TestWrapper<T>::unwrap() { return wrappedObj; }
template <typename T> T TestWrapper<T>::unwrap() const { return wrappedObj; }

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTWRAPPER_IMPL_HPP_
