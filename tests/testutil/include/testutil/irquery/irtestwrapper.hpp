// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_IR_TEST_WRAPPER_HPP
#define TEST_UTILS_IR_QUERY_IR_TEST_WRAPPER_HPP

#include "testutil/irquery/require.hpp"
#include "testutil/irquery/testwrapper.hpp"
#include <functional>
#include <popart/vendored/optional.hpp>

#include "popart/graphid.hpp"

namespace popart {
class Ir;

namespace irquery {
class GraphTestWrapper;

/**
 * This class wraps around an IR so as to make testing of it easier.
 *
 * NOTE: The set of available queries is incomplete at present. Feel free to add
 * whatever think would be useful, but please also add unit tests for any
 * queries you add.
 */
class IrTestWrapper : public TestWrapper<std::reference_wrapper<Ir>> {
public:
  /**
   * Constructor.
   **/
  explicit IrTestWrapper(Ir &ir);

  /**
   * NOTE: See comments on `Require` as to the intent of the `testReq` param.
   *
   * Express a query as to whether a graph with a specific ID exists in the
   * IR. If so, return a test wrapper for said graph, if not, return nullptr.
   * \param id The graph ID to look for.
   * \param testReq A parameter that defines when to generate a BOOST_REQUIRE
   *     failure, allowing for positive and negative testing (and no testing).
   * \return A test wrapper for the graph iff the graph was found.
   **/
  nonstd::optional<GraphTestWrapper>
  hasGraph(GraphId id, Require testReq = Require::Nothing);
};

} // namespace irquery
} // namespace popart

#endif
