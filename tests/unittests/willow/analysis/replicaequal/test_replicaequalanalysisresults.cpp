// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_ReplicaEqualAnalysis
#include <analysis/replicaequal/replicaequalanalysisresults.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <map>
#include <testutil/test_graphs/graph_test_models.hpp>
#include <vector>

#include "popart/graph.hpp"
#include "popart/graphid.hpp"
#include "popart/ir.hpp"
#include "popart/replicatedstreammode.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
class Op;
class internal_error;
} // namespace popart

using namespace popart;

namespace {

/**
 * Helper function to get schedules.
 */
ReplicaEqualAnalysisResults::GraphSchedules getGraphSchedules(Ir &ir) {
  std::map<GraphId, std::vector<Op *>> graphSchedules;
  for (auto graph : ir.getAllGraphs()) {
    graphSchedules[graph->id] =
        graph->getOpSchedule({}, RequireOptimalSchedule::No);
  }

  return graphSchedules;
}

} // namespace

using namespace popart;

BOOST_AUTO_TEST_CASE(ReplicaEqualsAnalysisResults_getAndSetValues) {

  ReplicaEqualAnalysisResults results;

  // Initialised based on test graph model.
  GraphTestModel4 model(ReplicatedStreamMode::Broadcast);
  auto &ir            = model.getIr();
  auto graphSchedules = getGraphSchedules(ir);
  results.setGraphSchedules(graphSchedules);

  auto &mainGraph        = ir.getMainGraph();
  auto mainGraphSchedule = graphSchedules.at(mainGraph.id);

  BOOST_REQUIRE_EQUAL(5, mainGraphSchedule.size());

  // NOTE: We just set/get tensors at arbitrary places, not places that make
  // sense for GraphTestModel4.
  auto w = mainGraph.getTensor("w");

  // Start with no results stored, check all `getValue{At,Before}` methods fail.

  // Check there is no value for w at init time yet.
  BOOST_REQUIRE_THROW(results.getValueAt(w, nonstd::nullopt),
                      popart::internal_error);

  // Check there is no value for w at time 2 yet.
  BOOST_REQUIRE_THROW(results.getValueAt(w, mainGraphSchedule[2]),
                      popart::internal_error);

  // Check there is no value for w before time 2 yet.
  BOOST_REQUIRE_THROW(results.getValueBefore(w, mainGraphSchedule[2]),
                      popart::internal_error);

  // Set w to true at init time. This should allow us to find the result for w
  // at init time and at any time after using `getValueBefore`.
  results.setValueAt(w, nonstd::nullopt, true);

  // Check w is set at init time to true.
  BOOST_REQUIRE_EQUAL(results.getValueAt(w, nonstd::nullopt), true);

  // Check there is no value for w at time 2 yet.
  BOOST_REQUIRE_THROW(results.getValueAt(w, mainGraphSchedule[2]),
                      popart::internal_error);

  // Check w is set before time 2 to true.
  BOOST_REQUIRE_EQUAL(results.getValueBefore(w, mainGraphSchedule[2]), true);

  // Set w to false at init time. The value at init time should change to false
  // because values are logically AND'ed.
  results.setValueAt(w, nonstd::nullopt, false);

  // Check w is set at init time to false.
  BOOST_REQUIRE_EQUAL(results.getValueAt(w, nonstd::nullopt), false);

  // Check there is no value for w at time 2 yet.
  BOOST_REQUIRE_THROW(results.getValueAt(w, mainGraphSchedule[2]),
                      popart::internal_error);

  // Check w is set before time 2 to false.
  BOOST_REQUIRE_EQUAL(results.getValueBefore(w, mainGraphSchedule[2]), false);

  // Now set it back to true -- it should remain false.
  results.setValueAt(w, nonstd::nullopt, true);

  // Check w is set at init time to false.
  BOOST_REQUIRE_EQUAL(results.getValueAt(w, nonstd::nullopt), false);

  // Check there is no value for w at time 2 yet.
  BOOST_REQUIRE_THROW(results.getValueAt(w, mainGraphSchedule[2]),
                      popart::internal_error);

  // Check w is set before time 2 to false.
  BOOST_REQUIRE_EQUAL(results.getValueBefore(w, mainGraphSchedule[2]), false);

  // Now set true at time 2.
  results.setValueAt(w, mainGraphSchedule[2], true);

  // The value of 'w' at init time should be untouched.
  BOOST_REQUIRE_EQUAL(results.getValueAt(w, nonstd::nullopt), false);

  // We now have a value at time 2.
  BOOST_REQUIRE_EQUAL(results.getValueAt(w, mainGraphSchedule[2]), true);

  // The value before 2 should be false, still.
  BOOST_REQUIRE_EQUAL(results.getValueBefore(w, mainGraphSchedule[2]), false);
}

BOOST_AUTO_TEST_CASE(ReplicaEqualsAnalysisResults_changeFlag) {
  ReplicaEqualAnalysisResults results;

  // Initialised based on test graph model.
  GraphTestModel4 model(ReplicatedStreamMode::Broadcast);
  auto &ir            = model.getIr();
  auto graphSchedules = getGraphSchedules(ir);
  results.setGraphSchedules(graphSchedules);

  auto &mainGraph        = ir.getMainGraph();
  auto mainGraphSchedule = graphSchedules.at(mainGraph.id);
  auto w                 = mainGraph.getTensor("w");

  BOOST_REQUIRE_EQUAL(5, mainGraphSchedule.size());

  // Nothing has been set, so no changes.
  BOOST_REQUIRE_EQUAL(false, results.hasChanged());

  // Set something to true and expect a change.
  results.setValueAt(w, nonstd::nullopt, true);
  BOOST_REQUIRE_EQUAL(true, results.hasChanged());

  // Clear the results and expect no changes.
  results.clearChanges();
  BOOST_REQUIRE_EQUAL(false, results.hasChanged());

  // Set the same thing and dont expect a change.
  results.setValueAt(w, nonstd::nullopt, true);
  BOOST_REQUIRE_EQUAL(false, results.hasChanged());

  // Set the same thing to false and expect a change.
  results.setValueAt(w, nonstd::nullopt, false);
  BOOST_REQUIRE_EQUAL(true, results.hasChanged());

  // Clear the results and expect no changes.
  results.clearChanges();
  BOOST_REQUIRE_EQUAL(false, results.hasChanged());

  // Set the same thing to true again and expect no changes because
  // AND(true,false)=false.
  results.setValueAt(w, nonstd::nullopt, true);
  BOOST_REQUIRE_EQUAL(false, results.hasChanged());

  // Now set something else to true and expect changes.
  results.setValueAt(w, mainGraphSchedule[2], true);
  BOOST_REQUIRE_EQUAL(true, results.hasChanged());
}

BOOST_AUTO_TEST_CASE(ReplicaEqualsAnalysisResults_disagreements) {
  ReplicaEqualAnalysisResults results;

  // Initialised based on test graph model.
  GraphTestModel4 model(ReplicatedStreamMode::Broadcast);
  auto &ir            = model.getIr();
  auto graphSchedules = getGraphSchedules(ir);
  results.setGraphSchedules(graphSchedules);

  auto &mainGraph        = ir.getMainGraph();
  auto mainGraphSchedule = graphSchedules.at(mainGraph.id);
  auto w                 = mainGraph.getTensor("w");

  BOOST_REQUIRE_EQUAL(5, mainGraphSchedule.size());

  // Nothing has been set - no disagreement.
  BOOST_REQUIRE_EQUAL(false, results.hasDisagreements());

  // Set something to true - no disagreement.
  results.setValueAt(w, nonstd::nullopt, true);
  BOOST_REQUIRE_EQUAL(false, results.hasDisagreements());

  // Set the same thing - no disagreement.
  results.setValueAt(w, nonstd::nullopt, true);
  BOOST_REQUIRE_EQUAL(false, results.hasDisagreements());

  // Set the same thing to false - no disagreement.
  results.setValueAt(w, nonstd::nullopt, false);
  BOOST_REQUIRE_EQUAL(false, results.hasDisagreements());

  // Set the same thing to true again - expect disagreement as something
  // we asked for wasn't actually set!
  results.setValueAt(w, nonstd::nullopt, true);
  BOOST_REQUIRE_EQUAL(true, results.hasDisagreements());

  // Check the actual list of disagreements.
  BOOST_REQUIRE_EQUAL(ReplicaEqualAnalysisResults::Disagreements({w}),
                      results.getDisagreements());

  // Clear the results and expect no changes.
  results.clearChanges();
  BOOST_REQUIRE_EQUAL(false, results.hasDisagreements());

  // Check the actual list of disagreements is now empty.
  BOOST_REQUIRE_EQUAL(ReplicaEqualAnalysisResults::Disagreements({}),
                      results.getDisagreements());

  // Now set something else to true - no disagreement.
  results.setValueAt(w, mainGraphSchedule[2], true);
  BOOST_REQUIRE_EQUAL(false, results.hasDisagreements());
}

BOOST_AUTO_TEST_CASE(ReplicaEqualsAnalysisResults_streamOperator) {
  ReplicaEqualAnalysisResults results;

  // Initialised based on test graph model.
  GraphTestModel4 model(ReplicatedStreamMode::Broadcast);
  auto &ir            = model.getIr();
  auto graphSchedules = getGraphSchedules(ir);
  results.setGraphSchedules(graphSchedules);

  auto &mainGraph        = ir.getMainGraph();
  auto mainGraphSchedule = graphSchedules.at(mainGraph.id);

  BOOST_REQUIRE_EQUAL(5, mainGraphSchedule.size());

  // NOTE: We just set/get tensors at arbitrary places, not places that make
  // sense for GraphTestModel4.
  auto w         = mainGraph.getTensor("w");
  auto x__t0__t1 = mainGraph.getTensor("x__t0__t1");

  // Now set true at time 2.
  results.setValueAt(w, nonstd::nullopt, false);
  results.setValueAt(x__t0__t1, nonstd::nullopt, false);
  results.setValueAt(x__t0__t1, mainGraphSchedule[2], true);
  results.setValueAt(x__t0__t1, mainGraphSchedule[4], true);

  // Check the stream output.
  std::cout << "Results: " << results << std::endl;

  std::stringstream ss;
  ss << results;
  BOOST_REQUIRE_EQUAL("'w': [initTime->0]\n"
                      "'x__t0__t1': [\n"
                      "  initTime->0\n"
                      "  102 (ai.graphcore.Accumulate:1)->1\n"
                      "  104 (ai.graphcore.HostStore:1)->1\n"
                      "]\n",
                      ss.str());
}
