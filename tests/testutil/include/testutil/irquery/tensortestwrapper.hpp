#ifndef TEST_UTILS_IR_QUERY_TENSOR_TEST_WRAPPER_HPP
#define TEST_UTILS_IR_QUERY_TENSOR_TEST_WRAPPER_HPP

// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op.hpp>

#include <popart/vendored/optional.hpp>

#include <testutil/irquery/opstestwrapper.hpp>
#include <testutil/irquery/require.hpp>
#include <testutil/irquery/testwrapper.hpp>

namespace popart {
namespace irquery {

/**
 * This class wraps around a Tensor so as to make testing of it easier.
 *
 * NOTE: The set of available queries is incomplete at present. Feel free to add
 * whatever think would be useful, but please also add unit tests for any
 * queries you add.
 */
class TensorTestWrapper : public TestWrapper<Tensor *> {
public:
  /**
   * Constructor.
   **/
  TensorTestWrapper(Ir &ir, Tensor *tensor);

  /**
   * Get a test wrapper object representing this tensor's consumers.
   * \return A OpsTestWrapper object that can be used in queries.
   */
  OpsTestWrapper consumers();
};

} // namespace irquery
} // namespace popart

#endif