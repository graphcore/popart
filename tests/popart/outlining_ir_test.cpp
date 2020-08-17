// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE OutliningIrTest

#include <boost/test/unit_test.hpp>
#include <string>
#include <test_runner.hpp>
#include <popart/builder.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

BOOST_AUTO_TEST_CASE(TestOutliningWithExtraAttributes) {
  auto test = [](int numOutliningContexts = 1) {
    TestRunner runner;
    runner.isTraining = true;
    int N             = 8;
    int B             = 8;
    int K             = 4;
    int size          = 100;

    runner.buildModel([&](auto &builder) {
      auto aiOnnx = builder.aiOnnxOpset9();
      TensorInfo inInfo{"FLOAT", std::vector<int64_t>{B, 1, size}};
      auto act = builder.addInputTensor(inInfo);
      // N layers
      for (int n = 0; n < N; ++n) {
        auto attribute = std::vector<std::string>{
            "context", std::to_string(n % numOutliningContexts)};
        TensorInfo wInfo{"FLOAT", std::vector<int64_t>{1, size, size}};
        std::vector<TestTensor> inputs;
        std::vector<TestTensor> outputs;
        std::vector<float> wData(wInfo.nelms(), 0);
        ConstVoidData wCVData{wData.data(), wInfo};
        auto w = builder.addInitializedInputTensor(wCVData);
        act = aiOnnx.matmul({act, w}, logging::format("CHECKOP_MM: [{}]", n));
        builder.addNodeAttribute(sOutlineAttribute, attribute, {act});
        builder.virtualGraph(act, n % 2);
        act = aiOnnx.relu({act}, logging::format("CHECKOP_RELU: [{}]", n));
        builder.addNodeAttribute(sOutlineAttribute, attribute, {act});
        builder.virtualGraph(act, n % 2);
      }

      auto loss = builder.aiGraphcoreOpset1().l1loss({act}, 0.1);
      builder.virtualGraph(loss, 2);

      // Enable outlining with no restrictions
      runner.opts.explicitRecomputation          = false;
      runner.opts.enableOutlining                = true;
      runner.opts.outlineThreshold               = 1.0;
      runner.opts.enableOutliningCopyCostPruning = false;
      runner.opts.virtualGraphMode               = VirtualGraphMode::Manual;
      runner.patterns = Patterns(PatternsLevel::Default);
      // Disable so that no false negatives (rhs vs. lhs inplace) exist
      runner.patterns.inplaceEnabled = false;
      runner.loss                    = loss;

      return act;
    });

    // Testing that the schedule is as expected for batch serialization:
    runner.checkIr([&](Ir &ir) {
      std::vector<Op *> schedule = ir.getMainGraph().getOpSchedule({});

      std::map<GraphId, size_t> numCallsToSubgraph;

      // Testing that the schedule is as expected for outlining contexts:
      for (size_t i = 0; i < schedule.size(); i++) {
        Op *op = schedule.at(i);
        for (auto subgraph : op->getCalledGraphs()) {
          ++numCallsToSubgraph[subgraph->id];
        }
      }

      BOOST_CHECK(numCallsToSubgraph.find(GraphId("_subgraph(0)")) !=
                  numCallsToSubgraph.end());
      BOOST_CHECK(numCallsToSubgraph.find(GraphId("_subgraph(1)")) !=
                  numCallsToSubgraph.end());
      BOOST_CHECK(numCallsToSubgraph.find(GraphId("_subgraph(2)")) !=
                  numCallsToSubgraph.end());
      BOOST_CHECK(numCallsToSubgraph.find(GraphId("_subgraph(3)")) !=
                  numCallsToSubgraph.end());
      BOOST_CHECK(numCallsToSubgraph.find(GraphId("_subgraph(4)")) !=
                  numCallsToSubgraph.end());
      for (auto &graphIdAndCount : numCallsToSubgraph) {
        logging::trace("Calls to subgraph: {} {}",
                       graphIdAndCount.first,
                       graphIdAndCount.second);
        switch (numOutliningContexts) {
        case 1:
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(0)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(1)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(2)") ||
                      graphIdAndCount.second == 1);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(3)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(4)") ||
                      graphIdAndCount.second == 3);
          break;
        case 3:
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(0)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(1)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(2)") ||
                      graphIdAndCount.second == 1);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(3)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(4)") ||
                      graphIdAndCount.second == 3);
          break;
        case 4:
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(0)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(1)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(2)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(3)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(4)") ||
                      graphIdAndCount.second == 1);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(5)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("_subgraph(6)") ||
                      graphIdAndCount.second == 3);
          break;
        }
      }
    });
  };
  test(1);
  test(3);
  test(4);
}
