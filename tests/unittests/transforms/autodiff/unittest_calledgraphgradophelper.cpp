// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/bwdgraphinfo.hpp"
#define BOOST_TEST_MODULE unittest_calledgraphgradophelper

#include <sstream>

#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/transforms/autodiff/calledgraphgradophelper.hpp>

using namespace popart;

namespace {

/**
 * Subclass of OpWithCalledGraphs so we can instantiate and test it.
 **/
class TestOpWithCalledGraphs : public Op {
public:
  TestOpWithCalledGraphs(const Op::Settings &settings)
      : Op(OperatorIdentifier("OpWithCalledGraphs", "TestOps", 1), settings),
        calledGraphGradOpHelper{this} {}

  // Not relevant for these tests.
  virtual float getSubgraphValue() const override {
    return getLowSubgraphValue();
  }

  // Pass on to `calledGraphGradOpHelper`
  virtual void setCalledSubgraphGradInfo(
      const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) override {
    calledGraphGradOpHelper.setCalledSubgraphGradInfo(calledGraphsGradInfo);
  }

  // Not relevant for these tests.
  std::unique_ptr<Op> clone() const override {
    return std::make_unique<TestOpWithCalledGraphs>(*this);
  }

  std::vector<const Graph *> getCalledGraphs() const override {
    return calledGraphs;
  }

  InIndex subgraphInToOpInIndex(SubgraphIndex subgraphIndex,
                                InIndex inIndex) override {
    // Set subgraphInToOpInIndexImpl to override test class's behaviour.
    return subgraphInToOpInIndexImpl(subgraphIndex, inIndex);
  }

  OutIndex subgraphOutToOpOutIndex(SubgraphIndex subgraphIndex,
                                   OutIndex outIndex) override {
    // Set subgraphInToOpInIndexImpl to override test class's behaviour.
    return subgraphOutToOpOutIndexImpl(subgraphIndex, outIndex);
  }

  // Test implementer can set this.
  std::vector<const Graph *> calledGraphs{};
  // Test implementer can set this.
  std::function<InIndex(SubgraphIndex, InIndex)> subgraphInToOpInIndexImpl =
      [](SubgraphIndex, InIndex i) -> InIndex {
    throw error("subgraphInToOpInIndexImpl not set");
  };
  // Test implementer can set this.
  std::function<OutIndex(SubgraphIndex, OutIndex)> subgraphOutToOpOutIndexImpl =
      [](SubgraphIndex, OutIndex i) -> OutIndex {
    throw error("subgraphOutToOpOutIndexImpl not set");
  };

  CalledGraphGradOpHelper calledGraphGradOpHelper;
};
} // namespace

BOOST_AUTO_TEST_CASE(
    calledgraphgradophelper_missing_setCalledSubgraphGradInfo) {

  // Check that if `setCalledSubgraphGradInfo` isn't called the helper functions
  // in this class throw an exception.

  Ir ir{};
  Graph &graph = ir.createGraph(GraphId("main"));

  TestOpWithCalledGraphs op{Op::Settings{graph, "OpWithCalledGraphs0"}};
  auto &calledGraphGradOpHelper = op.calledGraphGradOpHelper;

  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getBwdGraph(0),
                      std::runtime_error);
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getBwdGraphInfo(0),
                      std::runtime_error);
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getCalledGraphGradInInfo(
                          0, [](InIndex i) { return i; }),
                      std::runtime_error);
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getCalledGraphGradOutToNonGradIn(
                          0, [](OutIndex i) { return i; }),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(calledgraphgradophelper_getBwdGraph) {

  // Check getBwdGraph works as expected.

  Ir ir{};
  Graph &graph           = ir.createGraph(GraphId("main"));
  Graph &fwdCalledGraphA = ir.createGraph(GraphId("A"));
  Graph &fwdCalledGraphB = ir.createGraph(GraphId("B"));
  Graph &bwdCalledGraphA = ir.createGraph(GraphId("A_bwd"));
  Graph &bwdCalledGraphB = ir.createGraph(GraphId("B_bwd"));

  TestOpWithCalledGraphs op{Op::Settings{graph, "OpWithCalledGraphs0"}};
  auto &calledGraphGradOpHelper = op.calledGraphGradOpHelper;

  // Set called graphs.
  op.calledGraphs = {&fwdCalledGraphA, &fwdCalledGraphB};

  // Set called subgraph grad info.
  op.setCalledSubgraphGradInfo(
      {{fwdCalledGraphA.id, {bwdCalledGraphA.id, {}, {}}},
       {fwdCalledGraphB.id, {bwdCalledGraphB.id, {}, {}}}});

  // Check now helper function `bwdCalledGraph` returns the right bwd graph.
  BOOST_REQUIRE(&bwdCalledGraphA == &calledGraphGradOpHelper.getBwdGraph(0));
  BOOST_REQUIRE(&bwdCalledGraphB == &calledGraphGradOpHelper.getBwdGraph(1));
  // And it fails on an invalid subgraph index.
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getBwdGraph(17),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(calledgraphgradophelper_getBwdGraphInfo) {

  // Check getBwdGraphInfo works as expected.

  Ir ir{};
  Graph &graph           = ir.createGraph(GraphId("main"));
  Graph &fwdCalledGraphA = ir.createGraph(GraphId("A"));
  Graph &fwdCalledGraphB = ir.createGraph(GraphId("B"));
  Graph &bwdCalledGraphA = ir.createGraph(GraphId("A_bwd"));
  Graph &bwdCalledGraphB = ir.createGraph(GraphId("B_bwd"));
  BwdGraphInfo bwdGraphInfoA{bwdCalledGraphA.id, {}, {}};
  BwdGraphInfo bwdGraphInfoB{bwdCalledGraphB.id, {}, {}};

  TestOpWithCalledGraphs op{Op::Settings{graph, "OpWithCalledGraphs0"}};
  auto &calledGraphGradOpHelper = op.calledGraphGradOpHelper;

  // Set called graphs.
  op.calledGraphs = {&fwdCalledGraphA, &fwdCalledGraphB};

  // Check it fails if `setCalledSubgraphGradInfo` isn't called yet.
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getBwdGraphInfo(0),
                      std::runtime_error);

  // Set called subgraph grad info.
  op.setCalledSubgraphGradInfo({{fwdCalledGraphA.id, bwdGraphInfoA},
                                {fwdCalledGraphB.id, bwdGraphInfoB}});

  // Check now helper function `getBwdGraphInfo` returns the right info.
  BOOST_REQUIRE(bwdGraphInfoA == calledGraphGradOpHelper.getBwdGraphInfo(0));
  BOOST_REQUIRE(bwdGraphInfoB == calledGraphGradOpHelper.getBwdGraphInfo(1));
  // And it fails on an invalid subgraph index.
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getBwdGraphInfo(17),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(calledgraphgradophelper_getCalledGraphGradInInfo) {

  //               Column explanation
  //               ==================
  // .------------ BwdGraphInfo.expectedInputs entries.
  // | .---------- From bwdGraphInToGradOpInIndex parameter which, in this
  // | |           test, is: [](i) { return i + 4 }. This to facilitate some
  // | |           GradOps where the input indexing on the Op doesn't match the
  // | |           input indexing on the subgraph.
  // | | .-------- Index of associated fwdGraphId input or output.
  // | | | .------ From subgraphInToOpInIndex applied to fwdGraph input index,
  // | | | |       which, in this test, is: [](i) { return i + 1 } or
  // | | | |       or subgraphOutToOpOutIndex which, in this test is:
  // | | | |       [](i) { return i + 2 }, depending on whether the fwdGraph
  // | | | |       tensor is an input or output.
  // | | | | .---- From BwdGraphInfo.expectedInputs type value and whether
  // | | | | |     the fwdGraphId is an input or output.
  // | | | | |
  // | | | | '------------------------------------------------------.
  // | | | '---------------------------------------------.          |
  // | | '-----------------------------------.           |          |
  // | '-----------------------------.       |           |          |
  // |------------------.            |       |           |          |
  // |                  |            |       |           |          |
  // v                  v            v       v           v          |
  // bwdGraph inputs    bwdGraph     bwdOp   fwdGraph    fwdOp      v
  // (fwdGraphId,type)  #in          #in     #in/#out    #in/#out   type
  // =================  ========     =====   ========    =====      ====
  // (A/a_in3,Fwd)      0            4       3   -       4   -      In
  // (A/a_out0,Fwd)     1            5       -   0       -   2      Out
  // (A/a_in2,Fwd)      2            6       2   -       3   -      In
  // (A/a_out1,FwdGrad) 3            7       -   1       -   3      GradOut
  // (A/a_in0,Fwd)      4            8       0   -       1   -      In
  //                                 |                   \___/      |
  //                          .------'                     |        |
  //                          |  .-------------------------'        |
  //                          |  |  .-------------------------------'
  // Expected result          |  |  |
  // ===============          v  v  v
  // vector<GradInOutMapper>{{4, 4, In},
  //                         {5, 2, Out},
  //                         {6, 3, In},
  //                         {7, 3, GradOut},
  //                         {8, 1, In}
  // };

  Ir ir{};
  Graph &graph          = ir.createGraph(GraphId("main"));
  Graph &fwdCalledGraph = ir.createGraph(GraphId("A"));
  Graph &bwdCalledGraph = ir.createGraph(GraphId("A_bwd"));

  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in0"), tInfo);
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in1"), tInfo);
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in2"), tInfo);
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in3"), tInfo);
  fwdCalledGraph.getTensors().addActGrad(fwdCalledGraph.addScope("a_out0"));
  fwdCalledGraph.markAsOutput(fwdCalledGraph.addScope("a_out0"));
  fwdCalledGraph.getTensors().addActGrad(fwdCalledGraph.addScope("a_out1"));
  fwdCalledGraph.markAsOutput(fwdCalledGraph.addScope("a_out1"));

  bwdCalledGraph.addInput(bwdCalledGraph.addScope("a_bwd_in0"), tInfo);
  bwdCalledGraph.addInput(bwdCalledGraph.addScope("a_bwd_in1"), tInfo);
  bwdCalledGraph.addInput(bwdCalledGraph.addScope("a_bwd_in2"), tInfo);
  bwdCalledGraph.addInput(bwdCalledGraph.addScope("a_bwd_in3"), tInfo);
  bwdCalledGraph.addInput(bwdCalledGraph.addScope("a_bwd_in4"), tInfo);

  BwdGraphInfo bwdGraphInfo{
      bwdCalledGraph.id,
      {
          // Expected inputs.
          {fwdCalledGraph.addScope("a_in3"), ExpectedConnectionType::Fwd},
          {fwdCalledGraph.addScope("a_out0"), ExpectedConnectionType::Fwd},
          {fwdCalledGraph.addScope("a_in2"), ExpectedConnectionType::Fwd},
          {fwdCalledGraph.addScope("a_out1"), ExpectedConnectionType::FwdGrad},
          {fwdCalledGraph.addScope("a_in0"), ExpectedConnectionType::Fwd},
      },
      {}};

  auto subgraphInToOpInIndex = [](SubgraphIndex, InIndex i) -> InIndex {
    return i + 1;
  };

  auto subgraphOutToOpOutIndex = [](SubgraphIndex, InIndex i) -> InIndex {
    return i + 2;
  };

  auto bwdGraphInToGradOpInIndex = [](InIndex i) -> InIndex { return i + 4; };

  std::vector<GradInOutMapper> expectedResult{{4, 4, GradOpInType::In},
                                              {5, 2, GradOpInType::Out},
                                              {6, 3, GradOpInType::In},
                                              {7, 3, GradOpInType::GradOut},
                                              {8, 1, GradOpInType::In}};

  // Check getCalledGraphGradInInfo works as expected.
  TestOpWithCalledGraphs op{Op::Settings{graph, "OpWithCalledGraphs0"}};
  auto &calledGraphGradOpHelper = op.calledGraphGradOpHelper;

  // Set called graphs.
  op.calledGraphs                = {&fwdCalledGraph};
  op.subgraphInToOpInIndexImpl   = subgraphInToOpInIndex;
  op.subgraphOutToOpOutIndexImpl = subgraphOutToOpOutIndex;

  // Check it fails if `setCalledSubgraphGradInfo` isn't called yet.
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getCalledGraphGradInInfo(
                          0, bwdGraphInToGradOpInIndex),
                      std::runtime_error);

  // Set called subgraph grad info.
  op.setCalledSubgraphGradInfo({{fwdCalledGraph.id, bwdGraphInfo}});

  // Check now helper function `getCalledGraphGradInInfo` returns the right
  // info.
  BOOST_REQUIRE(expectedResult ==
                calledGraphGradOpHelper.getCalledGraphGradInInfo(
                    0, bwdGraphInToGradOpInIndex));

  // And it fails on an invalid subgraph index.
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getCalledGraphGradInInfo(
                          17, bwdGraphInToGradOpInIndex),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(calledgraphgradophelper_getCalledGraphGradInInfo_error_0) {

  // Test what happens if fwd tensor is not an input or output.
  Ir ir{};
  Graph &graph          = ir.createGraph(GraphId("main"));
  Graph &fwdCalledGraph = ir.createGraph(GraphId("A"));
  Graph &bwdCalledGraph = ir.createGraph(GraphId("A_bwd"));

  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in0"), tInfo);
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in1"), tInfo);
  fwdCalledGraph.getTensors().addActGrad(fwdCalledGraph.addScope("a_out0"));
  fwdCalledGraph.markAsOutput(fwdCalledGraph.addScope("a_out0"));
  fwdCalledGraph.getTensors().addActGrad(fwdCalledGraph.addScope("a_out1"));
  fwdCalledGraph.markAsOutput(fwdCalledGraph.addScope("a_out1"));

  bwdCalledGraph.addInput(bwdCalledGraph.addScope("a_bwd_in0"), tInfo);

  BwdGraphInfo bwdGraphInfo{bwdCalledGraph.id,
                            {
                                // Expected inputs.
                                {fwdCalledGraph.addScope("non-existing-tensor"),
                                 ExpectedConnectionType::Fwd},
                            },
                            {}};

  auto bwdGraphInToGradOpInIndex = [](InIndex i) -> InIndex { return i; };

  // Check getCalledGraphGradInInfo works as expected.
  TestOpWithCalledGraphs op{Op::Settings{graph, "OpWithCalledGraphs0"}};
  auto &calledGraphGradOpHelper = op.calledGraphGradOpHelper;

  // Set called graphs.
  op.calledGraphs              = {&fwdCalledGraph};
  op.subgraphInToOpInIndexImpl = [](SubgraphIndex, InIndex i) -> InIndex {
    return i;
  };
  op.subgraphOutToOpOutIndexImpl = [](SubgraphIndex, InIndex i) -> InIndex {
    return i;
  };

  // Set called subgraph grad info.
  op.setCalledSubgraphGradInfo({{fwdCalledGraph.id, bwdGraphInfo}});

  // Check the function fails because it can't map the input.
  BOOST_REQUIRE_THROW((calledGraphGradOpHelper.getCalledGraphGradInInfo(
                          0, [](InIndex i) -> InIndex { return i; })),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(calledgraphgradophelper_getCalledGraphGradOutToNonGradIn) {

  //               Column explanation
  //               ==================
  // .------------ BwdGraphInfo.expectedOutputs entries.
  // | .---------- From bwdGraphOutToGradOpOutIndex parameter which, in this
  // | |           test, is: [](i) { return i + 4 }. This to facilitate some
  // | |           GradOps where the output indexing on the Op doesn't match the
  // | |           output indexing on the subgraph.
  // | | .-------- Index of associated fwdGraphId input or output.
  // | | | .------ From subgraphToOpInIndex applied to fwdGraph input index,
  // | | | |       which, in this test, is: [](i) { return i + 1 } or
  // | | | |       or subgraphOutToOpOutIndex which, in this test is:
  // | | | |       [](i) { return i + 2 }, depending on whether the fwdGraph
  // | | | |       tensor is an input or output.
  // | | | |
  // | | | '---------------------------------------------.
  // | | '-----------------------------------.           |
  // | '-----------------------------.       |           |
  // |------------------.            |       |           |
  // |                  |            |       |           |
  // v                  v            v       v           v
  // bwdGraph outputs   bwdGraph     bwdOp   fwdGraph    fwdOp
  // (fwdGraphId,type)  #out         #out    #in/#out    #in/#out
  // =================  ========     =====   ========    =====
  // (A/a_in2,FwdGrad)  0            4       2   -       3   -
  // (A/a_in0,FwdGrad)  1            5       0   -       1   -
  // (A/a_in1,FwdGrad)  2            6       1   -       2   -
  //                                 |                   |
  //                          .------'                   |
  //                          |  .-----------------------'
  //                          |  |
  // Expected result          |  |
  // ===============          v  v
  //           map<int, int>{{4, 3},
  //                         {5, 1},
  //                         {6, 2}
  //                        };

  Ir ir{};
  Graph &graph          = ir.createGraph(GraphId("main"));
  Graph &fwdCalledGraph = ir.createGraph(GraphId("A"));
  Graph &bwdCalledGraph = ir.createGraph(GraphId("A_bwd"));

  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in0"), tInfo);
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in1"), tInfo);
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in2"), tInfo);

  bwdCalledGraph.getTensors().addActGrad(bwdCalledGraph.addScope("a_bwd_out0"));
  bwdCalledGraph.markAsOutput(bwdCalledGraph.addScope("a_bwd_out0"));
  bwdCalledGraph.getTensors().addActGrad(bwdCalledGraph.addScope("a_bwd_out1"));
  bwdCalledGraph.markAsOutput(bwdCalledGraph.addScope("a_bwd_out1"));
  bwdCalledGraph.getTensors().addActGrad(bwdCalledGraph.addScope("a_bwd_out2"));
  bwdCalledGraph.markAsOutput(bwdCalledGraph.addScope("a_bwd_out2"));

  BwdGraphInfo bwdGraphInfo{
      bwdCalledGraph.id,
      {},
      {// Expected inputs.
       {fwdCalledGraph.addScope("a_in2"), ExpectedConnectionType::FwdGrad},
       {fwdCalledGraph.addScope("a_in0"), ExpectedConnectionType::FwdGrad},
       {fwdCalledGraph.addScope("a_in1"), ExpectedConnectionType::FwdGrad}}};

  auto subgraphInToOpInIndex = [](SubgraphIndex, InIndex i) -> InIndex {
    return i + 1;
  };

  auto bwdGraphOutToGradOpOutIndex = [](InIndex i) -> InIndex { return i + 4; };

  std::map<int, int> expectedResult{{4, 3}, {5, 1}, {6, 2}};

  // Check getCalledGraphGradOutToNonGradIn works as expected.
  TestOpWithCalledGraphs op{Op::Settings{graph, "OpWithCalledGraphs0"}};
  auto &calledGraphGradOpHelper = op.calledGraphGradOpHelper;

  // Set called graphs.
  op.calledGraphs              = {&fwdCalledGraph};
  op.subgraphInToOpInIndexImpl = subgraphInToOpInIndex;

  // Check it fails if `setCalledSubgraphGradInfo` isn't called yet.
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getCalledGraphGradInInfo(
                          0, bwdGraphOutToGradOpOutIndex),
                      std::runtime_error);

  // Set called subgraph grad info.
  op.setCalledSubgraphGradInfo({{fwdCalledGraph.id, bwdGraphInfo}});

  // Check now helper function `getCalledGraphGradOutToNonGradIn` returns the
  // right info.
  BOOST_REQUIRE(expectedResult ==
                calledGraphGradOpHelper.getCalledGraphGradOutToNonGradIn(
                    0, bwdGraphOutToGradOpOutIndex));

  // And it fails on an invalid subgraph index.
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getCalledGraphGradOutToNonGradIn(
                          17, bwdGraphOutToGradOpOutIndex),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(
    calledgraphgradophelper_getCalledGraphGradOutToNonGradIn_error_0) {

  Ir ir{};
  Graph &graph          = ir.createGraph(GraphId("main"));
  Graph &fwdCalledGraph = ir.createGraph(GraphId("A"));
  Graph &bwdCalledGraph = ir.createGraph(GraphId("A_bwd"));

  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  fwdCalledGraph.addInput(fwdCalledGraph.addScope("a_in0"), tInfo);

  bwdCalledGraph.getTensors().addActGrad(bwdCalledGraph.addScope("a_bwd_out0"));
  bwdCalledGraph.markAsOutput(bwdCalledGraph.addScope("a_bwd_out0"));

  BwdGraphInfo bwdGraphInfo{bwdCalledGraph.id,
                            {},
                            {// Expected inputs.
                             {fwdCalledGraph.addScope("non-existent"),
                              ExpectedConnectionType::FwdGrad}}};

  auto subgraphInToOpInIndex = [](SubgraphIndex, InIndex i) -> InIndex {
    return i + 1;
  };

  auto bwdGraphOutToGradOpOutIndex = [](InIndex i) -> InIndex { return i + 4; };

  TestOpWithCalledGraphs op{Op::Settings{graph, "OpWithCalledGraphs0"}};
  auto &calledGraphGradOpHelper = op.calledGraphGradOpHelper;

  // Set called graphs.
  op.calledGraphs              = {&fwdCalledGraph};
  op.subgraphInToOpInIndexImpl = subgraphInToOpInIndex;

  // Check it fails if `getCalledGraphGradInInfo` isn't called yet.
  BOOST_REQUIRE_THROW(calledGraphGradOpHelper.getCalledGraphGradInInfo(
                          0, bwdGraphOutToGradOpOutIndex),
                      std::runtime_error);

  // Set called subgraph grad info.
  calledGraphGradOpHelper.setCalledSubgraphGradInfo(
      {{fwdCalledGraph.id, bwdGraphInfo}});

  // Check now helper function `getCalledGraphGradOutToNonGradIn` fails
  // relatively gracefully.
  BOOST_REQUIRE_THROW((calledGraphGradOpHelper.getCalledGraphGradOutToNonGradIn(
                          0, bwdGraphOutToGradOpOutIndex)),
                      std::runtime_error);
}