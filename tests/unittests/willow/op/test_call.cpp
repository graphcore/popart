// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Willow_Op_Call
#include <boost/test/unit_test.hpp>
#include <map>
#include <string>
#include <popart/ir.hpp>
#include <popart/op/call.hpp>
#include <popart/op/init.hpp>

#include "popart/datatype.hpp"
#include "popart/graph.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/graphid.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

using namespace popart;

/**
 * Check that you can setup a CallOp without connecting all of the subgraph's
 * outputs.
 */
BOOST_AUTO_TEST_CASE(TestCanNotConnectAllSubgraphOutputsToCallOp) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto &sg = ir.createGraph(GraphId("sg"));

  const auto ti = TensorInfo{DataType::FLOAT, Shape{4}};

  const auto sg_x = sg.addInput(ti);
  sg.markAsOutput(sg_x);

  const TensorId x = "x";
  g.createConnectedOp<InitOp>({},
                              {{InitOp::getOutIndex(), x}},
                              Onnx::CustomOperators::Init_1,
                              ti,
                              TensorType::ActGrad,
                              InitType::Zero,
                              Op::Settings{g, "Init"});

  // Connect no outputs and setup. There should be no error.
  BOOST_REQUIRE_NO_THROW(
      g.createConnectedOp<CallOp>({{sg.getInputIndex(sg_x), x}},
                                  {},
                                  Onnx::CustomOperators::Call_1,
                                  sg,
                                  Op::Settings{g, "Call"}) //
  );
}
