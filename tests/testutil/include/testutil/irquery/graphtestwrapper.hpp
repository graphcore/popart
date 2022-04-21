// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_GRAPH_TEST_WRAPPER_HPP
#define TEST_UTILS_IR_QUERY_GRAPH_TEST_WRAPPER_HPP

#include <popart/graph.hpp>

#include <testutil/irquery/opstestwrapper.hpp>
#include <testutil/irquery/require.hpp>
#include <testutil/irquery/tensorindexmaptestwrapper.hpp>
#include <testutil/irquery/testwrapper.hpp>

namespace popart {
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

#endif
