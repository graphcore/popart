#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TENSORTESTWRAPPER_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TENSORTESTWRAPPER_HPP_

// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "testutil/irquery/opstestwrapper.hpp"
#include "testutil/irquery/testwrapper.hpp"

namespace popart {
class Ir;
class Tensor;

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

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TENSORTESTWRAPPER_HPP_
