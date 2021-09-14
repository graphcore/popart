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

namespace popart {

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
  TensorId xSg = addScope(sg.getScope(), "x");
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

BOOST_AUTO_TEST_CASE(mainloops_throw_art_not_supported) {
  auto test = [](AnchorReturnType art) {
    Ir ir;
    Graph &graph = ir.getMainGraph();

    const TensorInfo info(DataType::FLOAT, Shape{1});
    float xHost[] = {1};

    graph.getTensors().addVarInit("x", info, static_cast<void *>(&xHost));

    auto dataFlow = DataFlow(1, {{"x", art}});
    ir.setDataFlow(dataFlow);

    MainLoops mainLoops;

    // Only AnchorReturnTypeId::All and AnchorReturnTypeId::Sum should work. Any
    // other AnchorReturnTypeId should throw an error.
    if (art.id() == AnchorReturnTypeId::All ||
        art.id() == AnchorReturnTypeId::Sum) {
      mainLoops.apply(graph);
    } else {
      // Error defined in MainLoops::apply() of mainloops.cpp.
      std::string msg = "[MainLoops] AnchorReturnType::" + art.str() +
                        " for Tensor \"x\" is unsupported when explicit main "
                        "loops are enabled.";
      BOOST_REQUIRE_EXCEPTION(
          mainLoops.apply(graph), error, checkErrorMsgFunc(msg));
    }
  };

  AnchorReturnType arts[] = {AnchorReturnType("All"),
                             AnchorReturnType("Sum"),
                             AnchorReturnType("Final")};
  for (const auto art : arts) {
    test(art);
  }
}

} // namespace popart
