// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE pipeline_unittest

#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/op/call.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/incrementmod.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/sgd0combo.hpp>
#include <popart/tensornames.hpp>
#include <popart/transforms/pipeline.hpp>

#include <popart/graphutils.hpp>

#include <testutil/test_graphs/graph_test_models.hpp>

using namespace popart;

namespace {

class TestOp : public Op {
public:
  TestOp(bool preLoss, bool recompute, const Op::Settings &_settings)
      : Op(OperatorIdentifier("TestOps", "TestOp", 1), _settings) {
    scheduledPreLoss = preLoss ? ScheduledPreLoss::Yes : ScheduledPreLoss::No;
    settings.recomputeType =
        recompute ? RecomputeType::Recompute : RecomputeType::Checkpoint;
  }
  std::unique_ptr<Op> clone() const final {
    return std::make_unique<TestOp>(*this);
  }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  void setup() final { outInfo(0) = inInfo(0); }
};

class CopyOp : public Op {
public:
  CopyOp(const Op::Settings &settings)
      : Op(OperatorIdentifier("TestOps", "CopyOp", 1), settings) {}

  bool isIpuCopyOp() const final { return true; }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<CopyOp>(*this);
  }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  void setup() final { outInfo(0) = inInfo(0); }
};

TensorId addStartTensor(Graph &g, Op::Settings settings) {
  TensorId t0 = g.getIr().createIntermediateTensorId("t0");
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  g.getTensors().addVarInit(t0, t0Info, static_cast<void *>(&t0Data));
  auto t0_c = g.getIr().createIntermediateTensorId(t0);
  g.createConnectedOp<CopyOp>({{0, t0}}, {{0, t0_c}}, settings);
  return t0_c;
}

} // namespace

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_0) {
  Ir ir;

  auto &g                = ir.getMainGraph();
  Op::Settings settings  = Op::Settings{g, "testOp"};
  settings.pipelineStage = 0;

  auto t0 = addStartTensor(g, settings);

  // PreLoss
  auto op1 = g.createConnectedOp<TestOp>(
      {{0, t0}}, {{0, "t1"}}, true, false, settings);
  auto op2 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t2"}}, true, true, settings);
  auto op3 = g.createConnectedOp<TestOp>(
      {{0, "t2"}}, {{0, "t3"}}, true, false, settings);
  auto op4 = g.createConnectedOp<TestOp>(
      {{0, "t3"}}, {{0, "t4"}}, true, true, settings);
  // PostLoss
  g.createConnectedOp<TestOp>({{0, "t4"}}, {{0, "t5"}}, false, false, settings);

  Pipeline::setFinalFwdStageRecomputation(g);
  BOOST_REQUIRE(op1->settings.recomputeType == RecomputeType::Checkpoint);
  // op2 should be unchanged
  BOOST_REQUIRE(op2->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(op3->settings.recomputeType == RecomputeType::Checkpoint);
  // op4 should be changed from Recompute to Checkpoint
  BOOST_REQUIRE(op4->settings.recomputeType == RecomputeType::Checkpoint);
}

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_1) {
  Ir ir;

  auto &g                = ir.getMainGraph();
  Op::Settings settings  = Op::Settings{g, "testOp"};
  settings.pipelineStage = 0;

  auto a0 = addStartTensor(g, settings);
  auto b0 = addStartTensor(g, settings);

  // PreLoss
  g.createConnectedOp<TestOp>({{0, a0}}, {{0, "a1"}}, true, false, settings);
  auto a_op2 = g.createConnectedOp<TestOp>(
      {{0, "a1"}}, {{0, "a3"}}, true, true, settings);

  g.createConnectedOp<TestOp>({{0, b0}}, {{0, "b1"}}, true, false, settings);
  auto b_op2 = g.createConnectedOp<TestOp>(
      {{0, "b1"}}, {{0, "b3"}}, true, true, settings);

  g.createConnectedOp<TestOp>(
      {{0, "a3"}, {1, "b3"}}, {{0, "c"}}, true, false, settings);
  // PostLoss
  g.createConnectedOp<TestOp>({{0, "c"}}, {{0, "c1"}}, false, false, settings);

  Pipeline::setFinalFwdStageRecomputation(g);

  BOOST_REQUIRE(a_op2->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(b_op2->settings.recomputeType == RecomputeType::Recompute);
}

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_fail) {
  Ir ir;

  auto &g                 = ir.getMainGraph();
  Op::Settings settings0  = Op::Settings{g, "testOp"};
  settings0.pipelineStage = 0;

  auto t0 = addStartTensor(g, settings0);
  g.createConnectedOp<TestOp>({{0, t0}}, {{0, "t1"}}, true, false, settings0);
  g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t2"}}, false, false, settings0);

  Op::Settings settings1  = Op::Settings{g, "testOp"};
  settings1.pipelineStage = 1;

  auto t3 = addStartTensor(g, settings1);
  g.createConnectedOp<TestOp>({{0, t3}}, {{0, "t4"}}, true, false, settings1);
  g.createConnectedOp<TestOp>(
      {{0, "t4"}}, {{0, "t5"}}, false, false, settings1);

  // More than one 'finalFwdStage' candidate
  BOOST_REQUIRE_THROW(Pipeline::setFinalFwdStageRecomputation(g),
                      internal_error);
}

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_nofrontier) {
  Ir ir;

  auto &g                = ir.getMainGraph();
  Op::Settings settings  = Op::Settings{g, "testOp"};
  settings.pipelineStage = 0;

  auto t0 = addStartTensor(g, settings);

  // PreLoss
  g.createConnectedOp<TestOp>({{0, t0}}, {{0, "t1"}}, true, true, settings);
  g.createConnectedOp<TestOp>({{0, "t1"}}, {{0, "t2"}}, true, true, settings);
  // PostLoss
  g.createConnectedOp<TestOp>({{0, "t2"}}, {{0, "t3"}}, false, false, settings);

  // All pre-loss ops are marked recompute, so the frontier (from which
  // checkpoint annotations are forward-propagated) is empty.
  // Note: a call to Pipeline::setFinalFwdStageRecomputation is only expected
  // after a call to Pipeline::setRecompute, which should guarantee a
  // non-empty frontier.
  BOOST_REQUIRE_THROW(Pipeline::setFinalFwdStageRecomputation(g),
                      internal_error);
}

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_nontopoorder) {
  Ir ir;

  auto &g                = ir.getMainGraph();
  Op::Settings settings  = Op::Settings{g, "testOp"};
  settings.pipelineStage = 0;

  auto t0 = addStartTensor(g, settings);

  // PreLoss
  auto op1 =
      g.createConnectedOp<TestOp>({{0, t0}}, {{0, "t1"}}, true, true, settings);
  auto op2 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "out"}}, true, true, settings);
  auto op3 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t3"}}, true, false, settings);
  op2->connectInTensor(1, "t3");
  auto op4 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t4"}}, true, true, settings);
  op3->connectInTensor(1, "t4");
  auto op5 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t5"}}, true, false, settings);
  op4->connectInTensor(1, "t5");
  // PostLoss
  g.createConnectedOp<TestOp>(
      {{0, "out"}}, {{0, "postLoss"}}, false, false, settings);

  Pipeline::setFinalFwdStageRecomputation(g);

  BOOST_REQUIRE(op1->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(op2->settings.recomputeType == RecomputeType::Checkpoint);
  BOOST_REQUIRE(op4->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(op3->settings.recomputeType == RecomputeType::Checkpoint);
  BOOST_REQUIRE(op5->settings.recomputeType == RecomputeType::Checkpoint);
}

// Check unroll in main graph
void explicitPipelineHelperTestCheckMainGraph(
    ExplicitPipelineTestModel1 &testModel,
    int numLayers,
    std::map<PipelineStage, std::string> &stageGraphMap,
    std::map<int, PipelineStage> &predicateIndexToStageMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p      = 0;
  preds[p++] = [](const Op *op) { return op->isConvertibleTo<LoopOp>(); };

  // Expect (2 * numLayers - 2) ramp-up steps
  for (int rampUpStep = 0; rampUpStep < (2 * numLayers - 2); ++rampUpStep) {
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<InitOp>(); };
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<HostLoadOp>(); };
    edges.insert({p - 2, p - 1});
    for (auto stage = 0; stage < rampUpStep + 1; ++stage) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<CallOp>(); };
      edges.insert({p - 2, p - 1});
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<IpuCopyOp>();
      };
      edges.insert({p - 2, p - 1});
    }
    // Edge from last IpuCopyOp to LoopOp
    edges.insert({p - 1, 0});
    // Record the CallOp before the LoopOp for every except the last pipeline
    // stage
    predicateIndexToStageMap[p - 2] = rampUpStep;
  }

  // Expect (2 * numLayers - 2) ramp-down steps
  for (int rampDownStep = 0; rampDownStep < (2 * numLayers - 2);
       ++rampDownStep) {
    for (auto stage = 0; stage < rampDownStep + 1; ++stage) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<CallOp>(); };
      if (stage == 0) {
        // Edge from LoopOp to CallOp
        edges.insert({0, p - 1});
      } else {
        // Edge from previous pipeline stage (IpuCopyOp)
        edges.insert({p - 2, p - 1});
      }
      if (stage < rampDownStep) {
        preds[p++] = [](const Op *op) {
          return op->isConvertibleTo<IpuCopyOp>();
        };
        edges.insert({p - 2, p - 1});
      }
    }
    if (rampDownStep == 0) {
      // Record the CallOp after the LoopOp for the last pipeline stage
      predicateIndexToStageMap[p - 1] = 2 * numLayers - 2;
    }
  }

  logging::trace(
      "Number of predicates: {}, number of edges: {}", p, edges.size());

  auto matches =
      graphutils::findMatchingOps(testModel.loopOp->getGraph(), preds, edges);

  // Due to permutations on IpuCopyOp, 16 matches (1 unique match)
  BOOST_REQUIRE_EQUAL(matches.size(), 16);

  // Get the GraphID of each pipeline stage
  for (auto indexAndStage : predicateIndexToStageMap) {
    stageGraphMap[indexAndStage.second] =
        dynamic_cast<CallOp *>(matches.front().at(indexAndStage.first))
            ->getCalledGraphIds()
            .front()
            .str();
    logging::trace("Stage: {} GraphId: {}",
                   indexAndStage.second,
                   stageGraphMap[indexAndStage.second]);
  }
}

// Check unroll in loop subgraph
void explicitPipelineHelperTestCheckLoopSubgraph(
    ExplicitPipelineTestModel1 &testModel,
    int numLayers,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p = 0;

  // Expect (2 * numLayers - 1) stages
  for (PipelineStage stage = 0; stage < (2 * numLayers - 1); ++stage) {
    if (stage == 0) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<InitOp>(); };
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<HostLoadOp>();
      };
      edges.insert({p - 2, p - 1});
      // Edge to the CallOp
      edges.insert({p - 1, p});
    }
    preds[p++] = [stage, &stageGraphMap](const Op *op) {
      return op->isConvertibleTo<CallOp>() &&
             op->getCalledGraphIds().front().str() == stageGraphMap.at(stage);
    };
    if (stage < (2 * numLayers - 2)) {
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<IpuCopyOp>();
      };
      edges.insert({p - 2, p - 1});
    }
  }

  logging::trace(
      "Number of predicates: {}, number of edges: {}", p, edges.size());

  auto matches = graphutils::findMatchingOps(
      testModel.loopOp->getCalledGraph(), preds, edges);

  // Due to permutations on IpuCopyOp, 2 matches (1 unique match)
  BOOST_REQUIRE_EQUAL(matches.size(), 2);
}

// Check pipeline stage 0 and 1
void explicitPipelineHelperTestCheckStage0And1(
    ExplicitPipelineTestModel1 &testModel,
    int numMatMulsPerLayer,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  for (auto &stage : std::set<PipelineStage>{0, 1}) {
    graphutils::OpPredMap preds;
    graphutils::Edges edges;

    int p = 0;
    for (int i = 0; i < numMatMulsPerLayer; ++i) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<MatMulOp>(); };
      if (i > 0) {
        edges.insert({p - 2, p - 1});
      }
    }

    auto matches = graphutils::findMatchingOps(
        testModel.getIr().getGraph(GraphId(stageGraphMap.at(stage))),
        preds,
        edges);

    BOOST_REQUIRE_EQUAL(matches.size(), 1);
  }
}

// Check pipeline stage 2 subgraph: Combined fwd & bwd
void explicitPipelineHelperTestCheckStage2(
    ExplicitPipelineTestModel1 &testModel,
    int numMatMulsPerLayer,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p = 0;
  for (int i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<MatMulOp>(); };
    if (i > 0) {
      edges.insert({p - 2, p - 1});
    }
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<AddOp>(); };
    edges.insert({p - 2, p - 1});
  }

  for (int i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<SGD0ComboOp>();
    };
  }

  auto matches = graphutils::findMatchingOps(
      testModel.getIr().getGraph(GraphId(stageGraphMap.at(2))), preds, edges);

  // Account for SGD0ComboOp permutation (1 unique match)
  BOOST_REQUIRE_EQUAL(matches.size(), 2);
}

// Check pipeline stage 3 & 4 subgraphs
void explicitPipelineHelperTestCheckStage3And4(
    ExplicitPipelineTestModel1 &testModel,
    int numMatMulsPerLayer,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  for (PipelineStage stage : std::set<PipelineStage>{3, 4}) {
    graphutils::OpPredMap preds;
    graphutils::Edges edges;

    int p = 0;

    for (int i = 0; i < numMatMulsPerLayer; ++i) {
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<SGD0ComboOp>();
      };
    }

    auto matches = graphutils::findMatchingOps(
        testModel.getIr().getGraph(GraphId(stageGraphMap.at(stage))),
        preds,
        edges);

    // Account for SGD0ComboOp permutation (1 unique match)
    BOOST_REQUIRE_EQUAL(matches.size(), 2);
  }
}

// Test the pipeline helper on a graph:
// - Outlines pipeline stages
// - Decomposes loop
BOOST_AUTO_TEST_CASE(ExplicitPipelineHelperTest) {
  int numLayers          = 3;
  int numMatMulsPerLayer = 2;

  ExplicitPipelineTestModel1 testModel(numLayers, numMatMulsPerLayer);

  // Model sets number of iterations to 2 * numLayers
  BOOST_REQUIRE_EQUAL(testModel.loopOp->getTripCountValue(), 2 * numLayers);

  // Ensure pipeline stages set correctly on IpuCopyOp
  Pipeline::checkOpsPipelineStage(testModel.loopOp->getCalledGraph());

  // Apply explicit pipeline helper
  // (will outline pipeline stages and decompose the loop)
  ExplicitPipelineHelper explicitPipeline(testModel.loopOp->getCalledGraph());
  explicitPipeline.createExplicitPipeline();

  // For debugging
  for (auto op :
       testModel.getIr().getOpSchedule({}, RequireOptimalSchedule::Yes)) {
    logging::trace("Op : {}:{}", op->getGraph().id, op->debugName());
  }
  testModel.getIr().dotCheckpoint(testModel.getIr(),
                                  "ExplicitPipelineHelperTest0");

  // Check resulting graph

  // After the backward pass, there are 5 stages (2 new backward pass only
  // stages) instead of just the 3 forward stages. This results in 4 unroll
  // steps and 2 remaining iterations.
  BOOST_REQUIRE_EQUAL(testModel.loopOp->getTripCountValue(),
                      2 * numLayers - (2 * numLayers - 2));

  // Map of PipelineStage to GraphId
  std::map<PipelineStage, std::string> stageGraphMap;
  std::map<int, PipelineStage> predicateIndexToStageMap;

  explicitPipelineHelperTestCheckMainGraph(
      testModel, numLayers, stageGraphMap, predicateIndexToStageMap);
  explicitPipelineHelperTestCheckLoopSubgraph(
      testModel, numLayers, stageGraphMap);
  explicitPipelineHelperTestCheckStage0And1(
      testModel, numMatMulsPerLayer, stageGraphMap);
  explicitPipelineHelperTestCheckStage2(
      testModel, numMatMulsPerLayer, stageGraphMap);
  explicitPipelineHelperTestCheckStage3And4(
      testModel, numMatMulsPerLayer, stageGraphMap);
}

// Check unroll in main graph
void explicitPipelineTestCheckMainGraph(
    ExplicitPipelineTestModel1 &testModel,
    int numLayers,
    std::map<PipelineStage, std::string> &stageGraphMap,
    std::map<int, PipelineStage> &predicateIndexToStageMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p      = 0;
  preds[p++] = [](const Op *op) { return op->isConvertibleTo<LoopOp>(); };

  // Expect (2 * numLayers - 2) ramp-up steps
  for (int rampUpStep = 0; rampUpStep < (2 * numLayers - 2); ++rampUpStep) {
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<InitOp>(); };
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<HostLoadOp>(); };
    edges.insert({p - 2, p - 1});
    for (auto stage = 0; stage < rampUpStep + 1; ++stage) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<CallOp>(); };
      edges.insert({p - 2, p - 1});
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<IpuCopyOp>();
      };
      edges.insert({p - 2, p - 1});
    }
    // Edge from last IpuCopyOp to LoopOp
    edges.insert({p - 1, 0});
    // Record the CallOp before the LoopOp for every except the last pipeline
    // stage
    predicateIndexToStageMap[p - 2] = rampUpStep;
  }

  // Expect (2 * numLayers - 2) ramp-down steps
  for (int rampDownStep = 0; rampDownStep < (2 * numLayers - 2);
       ++rampDownStep) {
    for (auto stage = 0; stage < rampDownStep + 1; ++stage) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<CallOp>(); };
      if (stage == 0) {
        // Edge from LoopOp to CallOp
        edges.insert({0, p - 1});
      } else {
        // Edge from previous pipeline stage (IpuCopyOp)
        edges.insert({p - 2, p - 1});
      }
      if (stage < rampDownStep) {
        preds[p++] = [](const Op *op) {
          return op->isConvertibleTo<IpuCopyOp>();
        };
        edges.insert({p - 2, p - 1});
      }
    }
    if (rampDownStep == 0) {
      // Record the CallOp after the LoopOp for the last pipeline stage
      predicateIndexToStageMap[p - 1] = 2 * numLayers - 2;
    }
  }

  logging::trace(
      "Number of predicates: {}, number of edges: {}", p, edges.size());

  auto matches =
      graphutils::findMatchingOps(testModel.loopOp->getGraph(), preds, edges);

  // Due to permutations on IpuCopyOp, 16 matches (1 unique match)
  BOOST_REQUIRE_EQUAL(matches.size(), 16);

  // Get the GraphID of each pipeline stage
  for (auto indexAndStage : predicateIndexToStageMap) {
    stageGraphMap[indexAndStage.second] =
        dynamic_cast<CallOp *>(matches.front().at(indexAndStage.first))
            ->getCalledGraphIds()
            .front()
            .str();
    logging::trace("Stage: {} GraphId: {}",
                   indexAndStage.second,
                   stageGraphMap[indexAndStage.second]);
  }
}

// Check unroll in subgraph
void explicitPipelineTestCheckLoopSubgraph(
    ExplicitPipelineTestModel1 &testModel,
    int numLayers,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p = 0;

  // Expect (2 * numLayers - 1) stages
  for (PipelineStage stage = 0; stage < (2 * numLayers - 1); ++stage) {
    if (stage == 0) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<InitOp>(); };
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<HostLoadOp>();
      };
      edges.insert({p - 2, p - 1});
      // Edge to the CallOp
      edges.insert({p - 1, p});
    }
    preds[p++] = [stage, &stageGraphMap](const Op *op) {
      return op->isConvertibleTo<CallOp>() &&
             op->getCalledGraphIds().front().str() == stageGraphMap.at(stage);
    };
    if (stage < (2 * numLayers - 2)) {
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<IpuCopyOp>();
      };
      edges.insert({p - 2, p - 1});
    }
  }

  logging::trace(
      "Number of predicates: {}, number of edges: {}", p, edges.size());

  auto matches = graphutils::findMatchingOps(
      testModel.loopOp->getCalledGraph(), preds, edges);

  // Due to permutations on IpuCopyOp, 2 matches (1 unique match)
  BOOST_REQUIRE_EQUAL(matches.size(), 2);
}

// Check pipeline stage 0
void explicitPipelineTestCheckStage0(
    ExplicitPipelineTestModel1 &testModel,
    int numMatMulsPerLayer,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p = 0;
  for (int i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<MatMulOp>(); };
    if (i > 0) {
      // Edge from previous MatMulOp
      edges.insert({p - 3, p - 1});
    }
    // Pipeline stash updates
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<DynamicUpdateInplaceOp>() &&
             op->inTensor(DynamicUpdateInplaceOp::getUpdateInIndex())
                 ->isGraphInput();
    };
    edges.insert({p - 2, p - 1});
  }

  // Additional stash of the input
  preds[p++] = [](const Op *op) {
    return op->isConvertibleTo<DynamicUpdateInplaceOp>() &&
           op->inTensor(DynamicUpdateInplaceOp::getUpdateInIndex())
               ->isGraphInput();
  };

  // numMatMulsPerLayer + 1 stash counter updates
  for (size_t i = 0; i < numMatMulsPerLayer + 1; ++i) {
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<IncrementModOp>() &&
             op->inTensor(IncrementModOp::getInIndex())->isGraphInput() &&
             op->outTensor(IncrementModOp::getOutIndex())->isGraphOutput();
    };
  }

  auto matches = graphutils::findMatchingOps(
      testModel.getIr().getGraph(GraphId(stageGraphMap.at(0))), preds, edges);

  // 3 due to IncrementModOp ambiguity
  BOOST_REQUIRE_EQUAL(matches.size(), 6);
}

// Check pipeline stage 1
void explicitPipelineTestCheckStage1(
    ExplicitPipelineTestModel1 &testModel,
    int numMatMulsPerLayer,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p = 0;
  for (int i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<MatMulOp>(); };
    if (i > 0) {
      // Edge from previous MatMulOp
      edges.insert({p - 2, p - 1});
    }
  }

  // Pipeline stash updates
  preds[p++] = [](const Op *op) {
    return op->isConvertibleTo<DynamicUpdateInplaceOp>() &&
           op->inTensor(DynamicUpdateInplaceOp::getUpdateInIndex())
               ->isGraphInput();
  };
  // One pipeline stash for the first MatMul
  edges.insert({p - 3, p - 1});

  // One pipeline stash from the previous pipeline stage
  preds[p++] = [](const Op *op) {
    return op->isConvertibleTo<DynamicUpdateInplaceOp>() &&
           op->inTensor(DynamicUpdateInplaceOp::getUpdateInIndex())
               ->isGraphInput() &&
           op->inTensor(DynamicUpdateInplaceOp::getInIndex())->isGraphInput();
  };

  // numMatMulsPerLayer stash counter updates
  for (size_t i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<IncrementModOp>() &&
             op->inTensor(IncrementModOp::getInIndex())->isGraphInput() &&
             op->outTensor(IncrementModOp::getOutIndex())->isGraphOutput();
    };
  }

  auto matches = graphutils::findMatchingOps(
      testModel.getIr().getGraph(GraphId(stageGraphMap.at(1))), preds, edges);

  // 2 due to IncrementModOp ambiguity
  BOOST_REQUIRE_EQUAL(matches.size(), 2);
}

// Check pipeline stage 2 subgraph: Combined fwd & bwd
void explicitPipelineTestCheckStage2(
    ExplicitPipelineTestModel1 &testModel,
    int numMatMulsPerLayer,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p = 0;
  for (int i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<MatMulOp>(); };
    if (i > 0) {
      edges.insert({p - 2, p - 1});
    }
    // Restore stash
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<DynamicSliceOp>() &&
             op->inTensor(DynamicSliceOp::getInIndex())->isGraphInput();
    };
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<AddOp>(); };
    edges.insert({p - 3, p - 1});
    edges.insert({p - 2, p - 1});
  }

  for (int i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<SGD0ComboOp>();
    };
  }

  // One pipeline stash for the gradient to stage 4
  preds[p++] = [](const Op *op) {
    return op->isConvertibleTo<DynamicUpdateInplaceOp>();
  };

  // numMatMulsPerLayer stash counter updates (numMatMulsPerLayer restore
  // counters, 1 stash counter)
  for (size_t i = 0; i < numMatMulsPerLayer + 1; ++i) {
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<IncrementModOp>() &&
             op->inTensor(IncrementModOp::getInIndex())->isGraphInput() &&
             op->outTensor(IncrementModOp::getOutIndex())->isGraphOutput();
    };
  }

  auto matches = graphutils::findMatchingOps(
      testModel.getIr().getGraph(GraphId(stageGraphMap.at(2))), preds, edges);

  // Account for SGD0ComboOp and IncrementModOp permutations (1 unique match)
  BOOST_REQUIRE_EQUAL(matches.size(), 12);
}

// Check pipeline stage 3 subgraph
void explicitPipelineTestCheckStage3(
    ExplicitPipelineTestModel1 &testModel,
    int numMatMulsPerLayer,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p = 0;

  for (int i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<SGD0ComboOp>();
    };
  }

  // numMatMulsPerLayer stash (restore) counter updates and DynamicSliceOps
  for (size_t i = 0; i < numMatMulsPerLayer; ++i) {
    // Restore stash
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<DynamicSliceOp>() &&
             op->inTensor(DynamicSliceOp::getInIndex())->isGraphInput();
    };
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<IncrementModOp>() &&
             op->inTensor(IncrementModOp::getInIndex())->isGraphInput() &&
             op->outTensor(IncrementModOp::getOutIndex())->isGraphOutput();
    };
  }

  auto matches = graphutils::findMatchingOps(
      testModel.getIr().getGraph(GraphId(stageGraphMap.at(3))), preds, edges);

  // Account for SGD0ComboOp, DynamicSliceOp and IncrementModOp permutation (1
  // unique match)
  BOOST_REQUIRE_EQUAL(matches.size(), 8);
}

// Check pipeline stage 4 subgraph
void explicitPipelineTestCheckStage4(
    ExplicitPipelineTestModel1 &testModel,
    int numMatMulsPerLayer,
    std::map<PipelineStage, std::string> &stageGraphMap) {
  graphutils::OpPredMap preds;
  graphutils::Edges edges;

  int p = 0;

  for (int i = 0; i < numMatMulsPerLayer; ++i) {
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<SGD0ComboOp>();
    };
  }

  // numMatMulsPerLayer + 1 stash (restore) counter updates and DynamicSliceOps
  for (size_t i = 0; i < numMatMulsPerLayer + 1; ++i) {
    // Restore stash
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<DynamicSliceOp>() &&
             op->inTensor(DynamicSliceOp::getInIndex())->isGraphInput();
    };
    preds[p++] = [](const Op *op) {
      return op->isConvertibleTo<IncrementModOp>() &&
             op->inTensor(IncrementModOp::getInIndex())->isGraphInput() &&
             op->outTensor(IncrementModOp::getOutIndex())->isGraphOutput();
    };
  }

  auto matches = graphutils::findMatchingOps(
      testModel.getIr().getGraph(GraphId(stageGraphMap.at(4))), preds, edges);

  // Account for SGD0ComboOp, DynamicSliceOp and IncrementModOp permutation (1
  // unique match)
  BOOST_REQUIRE_EQUAL(matches.size(), 72);
}

// Test the pipeline transformation on a graph:
// - Contiguates IPU copies
// - Adds pipeline stashes
// - Outlines pipeline stages
// - Decomposes loop
BOOST_AUTO_TEST_CASE(ExplicitPipelineTest) {
  int numLayers          = 3;
  int numMatMulsPerLayer = 2;

  ExplicitPipelineTestModel1 testModel(numLayers, numMatMulsPerLayer);

  // Model sets number of iterations to 2 * numLayers
  BOOST_REQUIRE_EQUAL(testModel.loopOp->getTripCountValue(), 2 * numLayers);

  // Apply explicit pipeline
  // (will add pipeline stashes, outline pipeline stages and decompose the loop)
  testModel.getIr().applyTransform(Pipeline::id(),
                                   testModel.loopOp->getCalledGraph());

  // For debugging
  for (auto op :
       testModel.getIr().getOpSchedule({}, RequireOptimalSchedule::Yes)) {
    logging::trace("Op : {}:{}", op->getGraph().id, op->debugName());
  }
  testModel.getIr().dotCheckpoint(testModel.getIr(), "ExplicitPipelineTest0");

  // Check resulting graph

  // After the backward pass, there are 5 stages (2 new backward pass only
  // stages) instead of just the 3 forward stages. This results in 4 unroll
  // steps and 2 remaining iterations.
  BOOST_REQUIRE_EQUAL(testModel.loopOp->getTripCountValue(),
                      2 * numLayers - (2 * numLayers - 2));

  // Map of PipelineStage to GraphId
  std::map<PipelineStage, std::string> stageGraphMap;
  std::map<int, PipelineStage> predicateIndexToStageMap;

  explicitPipelineTestCheckMainGraph(
      testModel, numLayers, stageGraphMap, predicateIndexToStageMap);
  explicitPipelineTestCheckLoopSubgraph(testModel, numLayers, stageGraphMap);
  explicitPipelineTestCheckStage0(testModel, numMatMulsPerLayer, stageGraphMap);
  explicitPipelineTestCheckStage1(testModel, numMatMulsPerLayer, stageGraphMap);
  explicitPipelineTestCheckStage2(testModel, numMatMulsPerLayer, stageGraphMap);
  explicitPipelineTestCheckStage3(testModel, numMatMulsPerLayer, stageGraphMap);
  explicitPipelineTestCheckStage4(testModel, numMatMulsPerLayer, stageGraphMap);
}
