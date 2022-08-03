// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE OutliningIrTest

#include "test_runner.hpp"

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/vendored/any.hpp"
#include "popart/voiddata.hpp"

std::map<GraphId, size_t> getNumCallsToSubgraph(Ir &ir) {
  std::map<GraphId, size_t> numCallsToSubgraph;

  // Testing that the schedule is as expected for outlining contexts:
  for (const auto &op_id : ir.getMainGraphOps()) {
    Op *op = op_id.second.get();
    for (auto subgraph : op->getCalledGraphs()) {
      ++numCallsToSubgraph[subgraph->id];
    }
  }
  return numCallsToSubgraph;
}

BOOST_AUTO_TEST_CASE(TestOutliningWithExtraAttributes) {
  auto test = [](int numOutliningContexts = 1) {
    TestRunner runner;
    runner.isTraining = true;
    int N             = 8;
    int B             = 8;
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
      runner.patterns.enableInPlace(false);
      runner.loss = loss;

      return act;
    });

    // Testing that the schedule is as expected for batch serialization:
    runner.checkIr([&](Ir &ir) {
      std::map<GraphId, size_t> numCallsToSubgraph = getNumCallsToSubgraph(ir);

      BOOST_CHECK(numCallsToSubgraph.find(GraphId("call_subgraph(0)")) !=
                  numCallsToSubgraph.end());
      BOOST_CHECK(numCallsToSubgraph.find(GraphId("call_subgraph(1)")) !=
                  numCallsToSubgraph.end());
      BOOST_CHECK(numCallsToSubgraph.find(GraphId("call_subgraph(2)")) !=
                  numCallsToSubgraph.end());
      BOOST_CHECK(numCallsToSubgraph.find(GraphId("call_subgraph(3)")) !=
                  numCallsToSubgraph.end());
      BOOST_CHECK(numCallsToSubgraph.find(GraphId("call_subgraph(4)")) !=
                  numCallsToSubgraph.end());
      for (auto &graphIdAndCount : numCallsToSubgraph) {
        logging::trace("Calls to subgraph: {} {}",
                       graphIdAndCount.first,
                       graphIdAndCount.second);
        switch (numOutliningContexts) {
        case 1:
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(0)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(1)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(2)") ||
                      graphIdAndCount.second == 1);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(3)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(4)") ||
                      graphIdAndCount.second == 3);
          break;
        case 3:
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(0)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(1)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(2)") ||
                      graphIdAndCount.second == 1);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(3)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(4)") ||
                      graphIdAndCount.second == 3);
          break;
        case 4:
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(0)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(1)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(2)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(3)") ||
                      graphIdAndCount.second == 2);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(4)") ||
                      graphIdAndCount.second == 1);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(5)") ||
                      graphIdAndCount.second == 4);
          BOOST_CHECK(graphIdAndCount.first != GraphId("call_subgraph(6)") ||
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

BOOST_AUTO_TEST_CASE(TestOutliningAcrossBoundaries) {

  enum AttributeToSet { None = 0, OutlineContext, ExecutionContext };

  int N    = 8;
  int B    = 8;
  int size = 100;
  // Test that, for certain op attributes, changes in these attributes prevents
  // outlining from outlining a group of ops.
  auto test = [&](AttributeToSet attributeToSet,
                  std::function<int(int)> partitioner,
                  int expectedNumberOfSubgraphs) {
    TestRunner runner;
    runner.isTraining = false;

    runner.buildModel([&](auto &builder) {
      auto aiOnnx = builder.aiOnnxOpset9();
      TensorInfo inInfo{"FLOAT", std::vector<int64_t>{B, 1, size}};
      auto act = builder.addInputTensor(inInfo);

      auto genAttrs = [&](auto &n) -> std::map<std::string, popart::any> {
        int partition = partitioner(n);
        switch (attributeToSet) {
        case OutlineContext: {
          auto value =
              std::vector<std::string>{"context", std::to_string(partition)};
          return {{sOutlineAttribute, value}};
        }
        case ExecutionContext: {
          int64_t value = 0ull;
          switch (partition) {
          case 0: {
            value =
                static_cast<int64_t>(ExecutionContext::WeightsFromHostFragment);
            break;
          }
          case 1: {
            value = static_cast<int64_t>(ExecutionContext::Normal);
            break;
          }
          case 2: {
            value =
                static_cast<int64_t>(ExecutionContext::AccumulateOuterFragment);
            break;
          }
          case 3: {
            value =
                static_cast<int64_t>(ExecutionContext::WeightsToHostFragment);
            break;
          }
          default:
            BOOST_ASSERT(
                false); // Can't test with paritions >3 for ExecutionContext.
          }
          return {{sExecutionContextAttribute, value}};
        }
        case None:
        default: {
          // do nothing.
          return {};
        }
        }
      };

      // N layers
      for (int n = 0; n < N; ++n) {
        // Note we change tensor sizes with period 4 to avoid outlining
        // per-iteration.
        TensorInfo wInfo{
            "FLOAT",
            std::vector<int64_t>{1, size + (n % 4), size + ((n + 1) % 4)}};
        std::vector<TestTensor> inputs;
        std::vector<TestTensor> outputs;
        std::vector<float> wData(wInfo.nelms(), 0);
        ConstVoidData wCVData{wData.data(), wInfo};
        auto w     = builder.addInitializedInputTensor(wCVData);
        auto attrs = genAttrs(n);
        act        = builder.customOp(Onnx::AiOnnx::OpSet9::MatMul,
                               9,
                               {act, w},
                               1,
                               attrs,
                               logging::format("CHECKOP_MM: [{}]", n))[0];
        act        = builder.customOp(Onnx::AiOnnx::OpSet9::Relu,
                               9,
                               {act},
                               1,
                               attrs,
                               logging::format("CHECKOP_RELU: [{}]", n))[0];
      }

      // Enable outlining with no restrictions
      runner.opts.explicitRecomputation          = false;
      runner.opts.enableOutlining                = true;
      runner.opts.outlineThreshold               = 1.0;
      runner.opts.enableOutliningCopyCostPruning = false;
      // runner.opts.virtualGraphMode               = VirtualGraphMode::Manual;
      runner.patterns = Patterns(PatternsLevel::Default);
      // Disable so that no false negatives (rhs vs. lhs inplace) exist
      runner.patterns.enableInPlace(false);

      return act;
    });

    // Testing that the schedule is as expected for batch serialization:
    runner.checkIr([&](Ir &ir) {
      std::map<GraphId, size_t> numCallsToSubgraph = getNumCallsToSubgraph(ir);

      BOOST_CHECK_EQUAL(expectedNumberOfSubgraphs, numCallsToSubgraph.size());
    });
  };

  // These functions parition the ops in two sets in different ways. Each set is
  // labelled with different attribute values (depending on other parameters to
  // test).
  auto all_false     = [](int n) -> int { return false; };
  auto split         = [&](int n) -> int { return n >= N / 2; };
  auto alternate     = [](int n) -> int { return n % 2 == 0; };
  auto alternate_two = [](int n) -> int { return (n >> 1) % 2 == 0; };
  auto increment_two = [](int n) -> int { return (n >> 1); };

  // In text below we describe the two ops within one iteration of the test
  // sequences of letters a b c d a b c d. Note that pairs of ops that are
  // equivalent with another iteration use the same letter (note there are 8
  // iterations of the main loop, and the logic repeats after 4 iterations). A
  // normal attribute change is shown with a single quote, e.g. a turns into a'.
  // A boundary attribute change is shown with an asterisk, e.g. a turns into
  // a*. The op a* is not equivalent to a but can be outlined by the same
  // subgraph. However, a and a* cannot exist within the same subgraph. Finally,
  // subgraphs are shown using square brackets, e.g. [a b].

  // When no boundaries are set, the outliner will outline one subgraph with
  // four ops.
  std::cout << "Checking base case (a b c d a b c d -> [a b c d ] [a b c d])"
            << std::endl;
  test(None, all_false, 1);

  std::cout
      << "Checking outline context (a b c d a b c d -> [a b c d] [a b c d])"
      << std::endl;
  test(OutlineContext, all_false, 1);
  std::cout
      << "Checking outline context (a b c d a' b' c' d' -> a b c d a' b' c' d')"
      << std::endl;
  test(OutlineContext, split, 0);
  std::cout << "Checking outline context (a b' c d' a b' c d' -> [a b' c d'] "
               "[a b' c d'])"
            << std::endl;
  test(OutlineContext, alternate, 1);
  std::cout << "Checking outline context (a b c' d' a b c' d' -> [a b c' d'] "
               "[a b c' d'])"
            << std::endl;
  test(OutlineContext, alternate_two, 1);

  std::cout
      << "Checking execution context (a b c d a b c d -> [a b c d] [a b c d])"
      << std::endl;
  test(ExecutionContext, all_false, 1);
  std::cout << "Checking execution context (a b c d a* b* c* d* -> [a b c d] "
               "[a b c d])"
            << std::endl;
  test(ExecutionContext, split, 1);
  std::cout << "Checking execution context (a b c* d* a b c* d* -> [a b][c d] "
               "[a b][c d])"
            << std::endl;
  test(ExecutionContext, increment_two, 2);
}
