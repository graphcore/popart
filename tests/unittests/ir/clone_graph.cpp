// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CloneGraphTests

#include <testutil/test_graphs/graph_test_models.hpp>

#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

using namespace popart;

namespace {

void compareConstants(const Graph &originalSubGraph,
                      const Graph &clonedSubGraph) {
  std::vector<TensorInfo> originalTensorInfo;
  std::vector<TensorInfo> clonedTensorInfo;

  std::vector<void *> originalTensorData;
  std::vector<void *> clonedTensorData;

  auto originalConsts =
      originalSubGraph.getTensors().getOfType(TensorType::Const);
  auto clonedConsts = clonedSubGraph.getTensors().getOfType(TensorType::Const);
  for (auto orignalGraphTensorIt = originalConsts.begin(),
            clonedGraphTensorIt  = clonedConsts.begin();
       orignalGraphTensorIt != originalConsts.end() ||
       clonedGraphTensorIt != clonedConsts.end();
       ++orignalGraphTensorIt, ++clonedGraphTensorIt) {
    originalTensorInfo.push_back((*orignalGraphTensorIt)->info);
    originalTensorData.push_back((*orignalGraphTensorIt)->tensorData()->data());

    clonedTensorInfo.push_back((*clonedGraphTensorIt)->info);
    clonedTensorData.push_back((*clonedGraphTensorIt)->tensorData()->data());
  }
  // Comparing tensorinfos
  BOOST_REQUIRE_EQUAL(originalTensorInfo, clonedTensorInfo);
  // Comparing pointers to the data
  BOOST_REQUIRE_EQUAL(originalTensorData, clonedTensorData);
}

void compareOpSchedules(const Graph &originalSubGraph,
                        const Graph &clonedSubGraph) {
  std::vector<std::string> originalOpNames;
  std::vector<std::string> clonedOpNames;

  auto originalScheduledOps =
      originalSubGraph.getOpSchedule({}, RequireOptimalSchedule::No);
  auto clonedScheduledOps =
      clonedSubGraph.getOpSchedule({}, RequireOptimalSchedule::No);

  for (auto originalOpIt = originalScheduledOps.begin(),
            clonedOpIt   = clonedScheduledOps.begin();
       originalOpIt != originalScheduledOps.end() ||
       clonedOpIt != clonedScheduledOps.end();
       ++originalOpIt, ++clonedOpIt) {
    originalOpNames.push_back((*originalOpIt)->name());
    clonedOpNames.push_back((*clonedOpIt)->name());
  }
  BOOST_CHECK_EQUAL(originalOpNames, clonedOpNames);
}

void checkClonedTensors(Ir &ir,
                        const std::string &cloneName,
                        const std::map<OpId, OpId> &originalOpIdAndClonedOpIds,
                        const std::vector<TensorId> &expectedSubGraphInput,
                        const std::vector<TensorId> &expectedSubGraphOutput) {
  auto &clonedSubGraph = ir.getGraph({cloneName});

  std::vector<TensorId> clonedSubGraphInput;
  std::vector<TensorId> clonedSubGraphOutput;

  auto opIds = clonedSubGraph.getOpIds();

  BOOST_CHECK_EQUAL(opIds.size() * 2, originalOpIdAndClonedOpIds.size());

  for (const auto originalOpIdAndClonedOpId : originalOpIdAndClonedOpIds) {
    if (std::find(opIds.begin(),
                  opIds.end(),
                  originalOpIdAndClonedOpId.second) != opIds.end()) {

      // auto subGraphOp = subGraph.getOp(originalOpIdAndClonedOpId.first);
      auto clonedSubGraphOp =
          clonedSubGraph.getOp(originalOpIdAndClonedOpId.second);

      auto tensorInputMap = clonedSubGraphOp->input->tensorMap();
      for (auto indexAndTensor : tensorInputMap) {
        clonedSubGraphInput.push_back(indexAndTensor.second->id);
      }

      auto tensorOutputMap = clonedSubGraphOp->output->tensorMap();
      for (auto indexAndTensor : tensorOutputMap) {
        clonedSubGraphOutput.push_back(indexAndTensor.second->id);
      }
    }
  }

  BOOST_CHECK_EQUAL(clonedSubGraphInput, expectedSubGraphInput);
  BOOST_CHECK_EQUAL(clonedSubGraphOutput, expectedSubGraphOutput);
}

} // namespace

BOOST_AUTO_TEST_CASE(TestCloneGraphTestModel1Sub0) {
  GraphTestModel1 model;
  auto &ir = model.getIr();

  // SubGraph 0
  std::string graphName0 = "sub0";
  auto cloneName0        = "clone" + graphName0;
  cloneName0[5]          = toupper(cloneName0[5]);

  auto originalOpIdAndClonedMap0 = ir.cloneGraph({graphName0}, {cloneName0});
  auto &originalSubGraph0        = ir.getGraph({graphName0});
  auto &clonedSubGraph0          = ir.getGraph({cloneName0});

  // Check constants
  compareConstants(originalSubGraph0, clonedSubGraph0);

  // Check opsSchedule
  compareOpSchedules(originalSubGraph0, clonedSubGraph0);

  // Check input and output tensor
  std::vector<TensorId> expectedInputSubGraph0{"cloneSub0/t3", "cloneSub0/t1"};
  std::vector<TensorId> expectedOutputSubGraph0{"cloneSub0/t7"};
  checkClonedTensors(ir,
                     cloneName0,
                     originalOpIdAndClonedMap0.opIdMap,
                     expectedInputSubGraph0,
                     expectedOutputSubGraph0);
}

BOOST_AUTO_TEST_CASE(TestCloneGraphTestModel1Sub1) {
  GraphTestModel1 model;
  auto &ir = model.getIr();

  // SubGraph 1
  std::string graphName1 = "sub1";
  auto cloneName1        = "clone" + graphName1;
  cloneName1[5]          = toupper(cloneName1[5]);

  auto originalOpIdAndClonedOpMap1 = ir.cloneGraph({graphName1}, {cloneName1});
  auto &originalSubGraph1          = ir.getGraph({graphName1});
  auto &clonedSubGraph1            = ir.getGraph({cloneName1});

  // Check constants
  compareConstants(originalSubGraph1, clonedSubGraph1);

  // Check opsSchedule
  compareOpSchedules(originalSubGraph1, clonedSubGraph1);

  // Check input and output tensor
  std::vector<TensorId> expectedInputSubGraph1{"cloneSub1/t2"};
  std::vector<TensorId> expectedOutputSubGraph1{"cloneSub1/t8"};
  checkClonedTensors(ir,
                     cloneName1,
                     originalOpIdAndClonedOpMap1.opIdMap,
                     expectedInputSubGraph1,
                     expectedOutputSubGraph1);
}
