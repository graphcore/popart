// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_OP_TEST_WRAPPER_IMPL_HPP
#define TEST_UTILS_IR_QUERY_OP_TEST_WRAPPER_IMPL_HPP

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

#endif
