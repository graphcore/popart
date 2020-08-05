#include <testutil/test_graphs/op/dummy.hpp>

#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/opidentifier.hpp>
#include <popart/scheduler.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

using namespace popart;

namespace test_graphs {

// So far don't see a use case for allowing the OperatorIdentifier and
// Op::Settings to be constructor parameters.
DummyOp::DummyOp(Graph &graph)
    : Op(test_graphs::CustomOperators::Dummy_1,
         Op::Settings{graph, "test_graph::DummyOp::Settings"}) {}

void DummyOp::setup() {
  if (nextInIndex == 0) {
    throw error("DummyOp::setup(): DummyOp requires at least one input.");
  }

  outInfo(getOutIndex()) = inInfo(0);
}

} // namespace test_graphs
