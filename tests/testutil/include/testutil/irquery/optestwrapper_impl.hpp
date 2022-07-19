// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_OPTESTWRAPPER_IMPL_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_OPTESTWRAPPER_IMPL_HPP_

#include <testutil/irquery/optestwrapper.hpp>
#include <testutil/irquery/tensorindexmaptestwrapper.hpp>

#include "testutil/irquery/testwrapper.hpp"

namespace popart {
class Ir;

namespace irquery {

template <typename OP, typename enableif>
OpTestWrapper<OP, enableif>::OpTestWrapper(Ir &ir, OP *op)
    : TestWrapper<OP *>{ir, op} {}

template <typename OP, typename enableif>
TensorIndexMapTestWrapper OpTestWrapper<OP, enableif>::inputs() {
  OP *op = TestWrapper<OP *>::wrappedObj;
  return TensorIndexMapTestWrapper{TestWrapper<OP *>::ir,
                                   op->input->tensorMap(),
                                   op->str(),
                                   "input",
                                   "inputs"};
}

template <typename OP, typename enableif>
TensorIndexMapTestWrapper OpTestWrapper<OP, enableif>::outputs() {
  OP *op = TestWrapper<OP *>::wrappedObj;
  return TensorIndexMapTestWrapper{TestWrapper<OP *>::ir,
                                   op->output->tensorMap(),
                                   op->str(),
                                   "output",
                                   "outputs"};
}

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_OPTESTWRAPPER_IMPL_HPP_
