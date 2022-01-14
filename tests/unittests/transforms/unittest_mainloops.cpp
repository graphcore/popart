// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MainLoopsUnittest
#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>

#include <memory>
#include <string>

#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/transforms/mainloops.hpp>
#include <popart/util.hpp>

using namespace popart;

std::function<bool(const error &)>
checkErrorMsgFunc(const std::string &prefix) {
  return [&](const error &ex) -> bool {
    return boost::algorithm::starts_with(ex.what(), prefix);
  };
}

BOOST_AUTO_TEST_CASE(mainloops_throw_anchor_in_subgraph) {
  auto ir   = std::make_unique<Ir>();
  Graph &g  = ir->getMainGraph();
  Graph &sg = ir->createGraph({"subgraph"});

  const TensorInfo info(DataType::FLOAT, Shape{1});
  float data[] = {1};

  TensorId x   = "x";
  TensorId xSg = addScope(sg, "x");
  g.getTensors().addVarInit(x, info, static_cast<void *>(&data));
  sg.addInput(xSg, info);

  ir->setDataFlow(DataFlow{1, {{x, AnchorReturnType("Sum")}}});
  ir->remapAnchor(x, xSg);

  MainLoops t;
  // Error defined in MainLoops::apply() of mainloops.cpp.
  std::string msg = "[MainLoops] The latest remap \"subgraph/x\" for anchor "
                    "\"x\" is not in the main graph.";
  BOOST_REQUIRE_EXCEPTION(t.apply(g), error, checkErrorMsgFunc(msg));
}
