// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_GRAPHTESTWRAPPER_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_GRAPHTESTWRAPPER_HPP_

#include "testutil/irquery/opstestwrapper.hpp"
#include "testutil/irquery/tensorindexmaptestwrapper.hpp"
#include "testutil/irquery/testwrapper.hpp"
#include <functional>

#include "popart/graphid.hpp"

namespace popart {
class Graph;
class Ir;

namespace irquery {

/**
 * This class wraps around a Graph so as to make testing of it easier.
 *
 * NOTE: The set of available queries is incomplete at present. Feel free to add
 * whatever think would be useful, but please also add unit tests for any
 * queries you add.
 */
class GraphTestWrapper : public TestWrapper<std::reference_wrapper<Graph>> {
public:
  /**
   * Constructor.
   **/
  GraphTestWrapper(Ir &ir, const GraphId &id);

  /**
   * Get a test wrapper object representing this graph's ops.
   * \return A OpsTestWrapper object that can be used in queries.
   */
  OpsTestWrapper ops();

  /**
   * Get a test wrapper object representing this graph's inputs.
   * \return A TensorIndexMapTestWrapper object that can be used in queries.
   */
  TensorIndexMapTestWrapper inputs();

  /**
   * Get a test wrapper object representing this graph's outputs.
   * \return A TensorIndexMapTestWrapper object that can be used in queries.
   */
  TensorIndexMapTestWrapper outputs();
};

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_GRAPHTESTWRAPPER_HPP_
