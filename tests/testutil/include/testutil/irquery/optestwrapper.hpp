// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_OPTESTWRAPPER_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_OPTESTWRAPPER_HPP_

#include "testutil/irquery/testwrapper.hpp"
#include <type_traits>
#include <popart/op.hpp>

namespace popart {
class Ir;

namespace irquery {

// Forward declaration.
class TensorIndexMapTestWrapper;

/**
 * This class wraps around an Op so as to make testing of it easier.
 *
 * NOTE: The set of available queries is incomplete at present. Feel free to add
 * whatever think would be useful, but please also add unit tests for any
 * queries you add.
 */
template <typename OP       = Op,
          typename enableif = std::enable_if_t<std::is_base_of<Op, OP>::value>>
class OpTestWrapper : public TestWrapper<OP *> {
public:
  /**
   * Constructor.
   **/
  OpTestWrapper(Ir &ir, OP *op);

  /**
   * Get a test wrapper object representing the op's inputs.
   * \return A TensorIndexMapTestWrapper object that can be used in queries.
   */
  TensorIndexMapTestWrapper inputs();

  /**
   * Get a test wrapper object representing the op's outputs.
   * \return A TensorIndexMapTestWrapper object that can be used in queries.
   */
  TensorIndexMapTestWrapper outputs();
};

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_OPTESTWRAPPER_HPP_
