// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AnchorReturnTypeValidationTest

#include <boost/test/unit_test.hpp>
#include <popart/dataflow.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/transforms/mainloops.hpp>

namespace popart {
bool hasPrefix(const std::string &str, const std::string &prefix) {
  return str.length() >= prefix.length() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

std::function<bool(const error &)>
checkErrorMsgFunc(const AnchorReturnType art) {
  return [&](const error &ex) -> bool {
    auto expectedPrefix = "AnchorReturnType::" + art.str() +
                          " for TensorId \"x\" is unsupported when explicit "
                          "main loops are enabled.";
    return hasPrefix(ex.what(), expectedPrefix);
  };
}

BOOST_AUTO_TEST_CASE(TestArtException) {
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
      BOOST_REQUIRE_EXCEPTION(
          mainLoops.apply(graph), error, checkErrorMsgFunc(art));
    }
  };

  AnchorReturnType arts[] = {AnchorReturnType("All"),
                             // TODO(T39577): Uncomment once this is resolved.
                             // AnchorReturnType("Sum"),
                             AnchorReturnType("Final")};
  for (const auto art : arts) {
    test(art);
  }
}
} // namespace popart
