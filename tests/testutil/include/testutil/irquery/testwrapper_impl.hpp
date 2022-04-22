// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_TEST_WRAPPER_IMPL_HPP
#define TEST_UTILS_IR_QUERY_TEST_WRAPPER_IMPL_HPP

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

#endif
