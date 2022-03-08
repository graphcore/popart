// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE decomposeloops_unittest

#include <boost/test/unit_test.hpp>

#include <popart/logging.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/matmul.hpp>

#include <popart/graphutils.hpp>
#include <popart/transforms/decomposeloops.hpp>

#include <testutil/irquery/irquery.hpp>
#include <testutil/test_graphs/graph_test_models.hpp>

using namespace popart;

// Test if the DecomposeLoopOverlapModel string representation is correct
BOOST_AUTO_TEST_CASE(DecomposeLoopOverlapModelStringRepresentationCheck) {

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  DecomposeLoopOverlapModel model(DecomposeTopoConLevel::Full,
                                  DecomposeTopoConLevel::Full,
                                  DecomposeTopoConLevel::Full,
                                  computeLikeExchangeStrategies);
  std::string modelString = model.getModelString();

  // See decomposeloops.hpp. The string representation visualizes how each
  // iteration of every type of operation will be unrolled and scheduled.
  // This is useful to check if the DecomposeModel does what is intended.
  std::string refString = "\
AuxiliaryBefore     0..1....|2......|......\n\
IoBeforeCompute     .0..1...|..2....|......\n\
IoToCompute         ..0...1.|.....2.|......\n\
Compute             .....0..|...1...|.2....\n\
ComputeToIo         .......0|......1|...2..\n\
IoAfterCompute      ........|.0.....|1...2.\n\
AuxiliaryAfter      ........|....0..|..1..2\n";

  logging::info("\n{}", modelString);

  BOOST_REQUIRE_EQUAL(modelString, refString);
}

// Test if the DecomposeLoopPipelineModel string representation is correct
BOOST_AUTO_TEST_CASE(DecomposeLoopPipelineModelStringRepresentationCheck) {
  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  {
    DecomposeLoopPipelineModel model(0,
                                     2,
                                     2,
                                     DecomposeTopoConLevel::Full,
                                     DecomposeTopoConLevel::Full,
                                     DecomposeTopoConLevel::Full,
                                     computeLikeExchangeStrategies);
    std::string modelString = model.getModelString();

    // See decomposeloops.hpp. The string representation visualizes how each
    // iteration of every type of operation will be unrolled and scheduled.
    // This is useful to check if the DecomposeModel does what is intended.
    std::string refString = "\
(PipelineStage: 0, type: AuxiliaryBefore, isPipelineIpuCopy: 0, isComputeLike: 0)     0.........1.........2.......|..3.......|...........................\n\
(PipelineStage: 0, type: IoBeforeCompute, isPipelineIpuCopy: 0, isComputeLike: 0)     .0.........1.........2......|...3......|...........................\n\
(PipelineStage: 0, type: IoToCompute, isPipelineIpuCopy: 0, isComputeLike: 0)         .....0.........1.........2..|.......3..|...........................\n\
(PipelineStage: 0, type: AuxiliaryBefore, isPipelineIpuCopy: 0, isComputeLike: 1)     ........0.........1.........|2.........|3..........................\n\
(PipelineStage: 0, type: IoBeforeCompute, isPipelineIpuCopy: 0, isComputeLike: 1)     .........0.........1........|.2........|.3.........................\n\
(PipelineStage: 0, type: Compute, isPipelineIpuCopy: 0, isComputeLike: 0)             ............0.........1.....|....2.....|....3......................\n\
(PipelineStage: 0, type: IoAfterCompute, isPipelineIpuCopy: 0, isComputeLike: 1)      .............0.........1....|.....2....|.....3.....................\n\
(PipelineStage: 0, type: AuxiliaryAfter, isPipelineIpuCopy: 0, isComputeLike: 1)      ..............0.........1...|......2...|......3....................\n\
(PipelineStage: 0, type: ComputeToIo, isPipelineIpuCopy: 0, isComputeLike: 0)         ................0.........1.|........2.|........3..................\n\
(PipelineStage: 0, type: Compute, isPipelineIpuCopy: 1, isComputeLike: 0)             .................0.........1|.........2|.........3.................\n\
(PipelineStage: 0, type: IoAfterCompute, isPipelineIpuCopy: 0, isComputeLike: 0)      .....................0......|...1......|...2.........3.............\n\
(PipelineStage: 0, type: AuxiliaryAfter, isPipelineIpuCopy: 0, isComputeLike: 0)      ........................0...|......1...|......2.........3..........\n\
                                                                                      ----------------------------+----------+---------------------------\n\
(PipelineStage: 1, type: AuxiliaryBefore, isPipelineIpuCopy: 0, isComputeLike: 0)     ..........0.........1.......|..2.......|..3........................\n\
(PipelineStage: 1, type: IoBeforeCompute, isPipelineIpuCopy: 0, isComputeLike: 0)     ...........0.........1......|...2......|...3.......................\n\
(PipelineStage: 1, type: IoToCompute, isPipelineIpuCopy: 0, isComputeLike: 0)         ...............0.........1..|.......2..|.......3...................\n\
(PipelineStage: 1, type: AuxiliaryBefore, isPipelineIpuCopy: 0, isComputeLike: 1)     ..................0.........|1.........|2.........3................\n\
(PipelineStage: 1, type: IoBeforeCompute, isPipelineIpuCopy: 0, isComputeLike: 1)     ...................0........|.1........|.2.........3...............\n\
(PipelineStage: 1, type: Compute, isPipelineIpuCopy: 0, isComputeLike: 0)             ......................0.....|....1.....|....2.........3............\n\
(PipelineStage: 1, type: IoAfterCompute, isPipelineIpuCopy: 0, isComputeLike: 1)      .......................0....|.....1....|.....2.........3...........\n\
(PipelineStage: 1, type: AuxiliaryAfter, isPipelineIpuCopy: 0, isComputeLike: 1)      ........................0...|......1...|......2.........3..........\n\
(PipelineStage: 1, type: ComputeToIo, isPipelineIpuCopy: 0, isComputeLike: 0)         ..........................0.|........1.|........2.........3........\n\
(PipelineStage: 1, type: Compute, isPipelineIpuCopy: 1, isComputeLike: 0)             ...........................0|.........1|.........2.........3.......\n\
(PipelineStage: 1, type: IoAfterCompute, isPipelineIpuCopy: 0, isComputeLike: 0)      ............................|...0......|...1.........2.........3...\n\
(PipelineStage: 1, type: AuxiliaryAfter, isPipelineIpuCopy: 0, isComputeLike: 0)      ............................|......0...|......1.........2.........3\n";

    logging::info("\n{}", modelString);

    BOOST_REQUIRE_EQUAL(modelString, refString);
  }

  {
    DecomposeLoopPipelineModel model(1,
                                     1,
                                     2,
                                     DecomposeTopoConLevel::Full,
                                     DecomposeTopoConLevel::Full,
                                     DecomposeTopoConLevel::Full,
                                     computeLikeExchangeStrategies);
    std::string modelString = model.getModelString();

    // See decomposeloops.hpp. The string representation visualizes how each
    // iteration of every type of operation will be unrolled and scheduled.
    // This is useful to check if the DecomposeModel does what is intended.
    std::string refString = "\
(PipelineStage: 0, type: AuxiliaryBefore, isPipelineIpuCopy: 0, isComputeLike: 1)     0.........|1.........|..........\n\
(PipelineStage: 0, type: IoBeforeCompute, isPipelineIpuCopy: 0, isComputeLike: 1)     .0........|.1........|..........\n\
(PipelineStage: 0, type: Compute, isPipelineIpuCopy: 0, isComputeLike: 0)             ....0.....|....1.....|..........\n\
(PipelineStage: 0, type: IoAfterCompute, isPipelineIpuCopy: 0, isComputeLike: 1)      .....0....|.....1....|..........\n\
(PipelineStage: 0, type: AuxiliaryAfter, isPipelineIpuCopy: 0, isComputeLike: 1)      ......0...|......1...|..........\n\
(PipelineStage: 0, type: ComputeToIo, isPipelineIpuCopy: 0, isComputeLike: 0)         ........0.|........1.|..........\n\
(PipelineStage: 0, type: Compute, isPipelineIpuCopy: 1, isComputeLike: 0)             .........0|.........1|..........\n\
(PipelineStage: 0, type: IoAfterCompute, isPipelineIpuCopy: 0, isComputeLike: 0)      ..........|...0......|...1......\n\
(PipelineStage: 0, type: AuxiliaryAfter, isPipelineIpuCopy: 0, isComputeLike: 0)      ..........|......0...|......1...\n\
                                                                                      ----------+----------+----------\n\
(PipelineStage: 1, type: AuxiliaryBefore, isPipelineIpuCopy: 0, isComputeLike: 0)     ..0.......|..1.......|..........\n\
(PipelineStage: 1, type: IoBeforeCompute, isPipelineIpuCopy: 0, isComputeLike: 0)     ...0......|...1......|..........\n\
(PipelineStage: 1, type: IoToCompute, isPipelineIpuCopy: 0, isComputeLike: 0)         .......0..|.......1..|..........\n\
(PipelineStage: 1, type: AuxiliaryBefore, isPipelineIpuCopy: 0, isComputeLike: 1)     ..........|0.........|1.........\n\
(PipelineStage: 1, type: IoBeforeCompute, isPipelineIpuCopy: 0, isComputeLike: 1)     ..........|.0........|.1........\n\
(PipelineStage: 1, type: Compute, isPipelineIpuCopy: 0, isComputeLike: 0)             ..........|....0.....|....1.....\n\
(PipelineStage: 1, type: IoAfterCompute, isPipelineIpuCopy: 0, isComputeLike: 1)      ..........|.....0....|.....1....\n\
(PipelineStage: 1, type: AuxiliaryAfter, isPipelineIpuCopy: 0, isComputeLike: 1)      ..........|......0...|......1...\n\
(PipelineStage: 1, type: ComputeToIo, isPipelineIpuCopy: 0, isComputeLike: 0)         ..........|........0.|........1.\n\
(PipelineStage: 1, type: Compute, isPipelineIpuCopy: 1, isComputeLike: 0)             ..........|.........0|.........1\n";

    logging::info("\n{}", modelString);

    BOOST_REQUIRE_EQUAL(modelString, refString);
  }
}

// Test Op classification with the IO overlap model, where:
// - All inputs are through compute tiles
// - All outputs are through compute tiles
BOOST_AUTO_TEST_CASE(DecomposeLoopOverlapClassifyTest) {
  int numStages   = 3;
  int numParallel = 2;

  ExplicitPipelineTestModel0 testModel(numStages, numParallel, {}, {});

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  // Unroll skewed by pipeline stage such that the main loop ends up
  // overlapping all pipeline stages
  DecomposeTopoConLevel before = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel loop   = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel after  = DecomposeTopoConLevel::Full;

  DecomposeLoopOverlapModel decomposeModel(
      before, loop, after, computeLikeExchangeStrategies);

  auto opTypeMap = decomposeModel.classifyOperations(*testModel.subgraphPt);

  for (auto &opType : opTypeMap) {
    logging::trace(
        "Op: {}, type: {}", opType.first->debugName(), opType.second);
    BOOST_REQUIRE_EQUAL(
        opType.second,
        DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::Compute});
  }
}

// Test Op classification with the IO overlap model, where:
// - All inputs are through IO tiles
//    (1 with ExchangeStrategy::OverlapInnerLoop)
//    (1 with ExchangeStrategy::JustInTime)
// - All outputs are through IO tiles
//    (1 with ExchangeStrategy::OverlapInnerLoop)
//    (1 with ExchangeStrategy::JustInTime)
BOOST_AUTO_TEST_CASE(DecomposeLoopOverlapClassifyTestOverlap) {
  int numStages   = 3;
  int numParallel = 2;

  // Build test model
  ExplicitPipelineTestModel0 testModel(
      numStages,
      numParallel,
      {{0, {TileSet::IO, ExchangeStrategy::OverlapInnerLoop}},
       {1, {TileSet::IO, ExchangeStrategy::JustInTime}}},
      {{0, {"all", TileSet::IO, ExchangeStrategy::OverlapInnerLoop}},
       {1, {"all", TileSet::IO, ExchangeStrategy::JustInTime}}});

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  // Unroll skewed by pipeline stage such that the main loop ends up
  // overlapping all pipeline stages
  DecomposeTopoConLevel before = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel loop   = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel after  = DecomposeTopoConLevel::Full;

  DecomposeLoopOverlapModel decomposeModel(
      before, loop, after, computeLikeExchangeStrategies);

  auto opTypeMap = decomposeModel.classifyOperations(*testModel.subgraphPt);

  for (auto &opType : opTypeMap) {
    logging::trace(
        "Op: {}, type: {}", opType.first->debugName(), opType.second);
  }

  auto &subgraph = *testModel.subgraphPt;

  // Build predicates and edges to check if the graph and classifications are
  // as expected.

  // Number of predicates: We expect 11 ops per path, based on the testModel.
  graphutils::OpPreds preds(numParallel * 11,
                            [](const Op *op) { return false; });
  graphutils::Edges edges;

  int p = 0;
  for (int i = 0; i < numParallel; ++i) {
    preds[p++] = [i, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<InitOp>() &&
             op->settings.tileSet == TileSet::IO &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 DecomposeLoopOpIOOverlapType{
                     i == 0 ? DecomposeLoopOpTypeEnum::AuxiliaryBefore
                            : DecomposeLoopOpTypeEnum::Compute};
    };
    preds[p++] = [i, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<HostLoadOp>() &&
             op->settings.tileSet == TileSet::IO &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 DecomposeLoopOpIOOverlapType{
                     i == 0 ? DecomposeLoopOpTypeEnum::IoBeforeCompute
                            : DecomposeLoopOpTypeEnum::Compute};
    };
    // Insert an edge between the previous two predicates
    edges.insert({p - 2, p - 1});
    preds[p++] = [i, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<IoTileCopyOp>() &&
             op->settings.tileSet == TileSet::Compute &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 DecomposeLoopOpIOOverlapType{
                     i == 0 ? DecomposeLoopOpTypeEnum::IoToCompute
                            : DecomposeLoopOpTypeEnum::Compute};
    };
    edges.insert({p - 2, p - 1});
    for (int j = 0; j < numStages; ++j) {
      preds[p++] = [&opTypeMap](const Op *op) {
        return op->isConvertibleTo<MatMulOp>() &&
               op->settings.tileSet == TileSet::Compute &&
               opTypeMap.at(op->getGraph().getOp(op->id)) ==
                   DecomposeLoopOpIOOverlapType{
                       DecomposeLoopOpTypeEnum::Compute};
      };
      edges.insert({p - 2, p - 1});
      if (j < numStages - 1) {
        preds[p++] = [&opTypeMap](const Op *op) {
          return op->isConvertibleTo<IpuCopyOp>() &&
                 op->settings.tileSet == TileSet::Compute &&
                 opTypeMap.at(op->getGraph().getOp(op->id)) ==
                     DecomposeLoopOpIOOverlapType{
                         DecomposeLoopOpTypeEnum::Compute};
        };
        edges.insert({p - 2, p - 1});
      }
    }
    preds[p++] = [&opTypeMap](const Op *op) {
      return op->isConvertibleTo<AccumulateOp>() &&
             op->settings.tileSet == TileSet::Compute &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 DecomposeLoopOpIOOverlapType{DecomposeLoopOpTypeEnum::Compute};
    };
    edges.insert({p - 2, p - 1});
    preds[p++] = [i, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<IoTileCopyOp>() &&
             op->settings.tileSet == TileSet::IO &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 DecomposeLoopOpIOOverlapType{
                     i == 0 ? DecomposeLoopOpTypeEnum::ComputeToIo
                            : DecomposeLoopOpTypeEnum::Compute};
    };
    edges.insert({p - 3, p - 1});
    preds[p++] = [i, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<HostStoreOp>() &&
             op->settings.tileSet == TileSet::IO &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 DecomposeLoopOpIOOverlapType{
                     i == 0 ? DecomposeLoopOpTypeEnum::IoAfterCompute
                            : DecomposeLoopOpTypeEnum::Compute};
    };
    edges.insert({p - 2, p - 1});
  }
  logging::trace(
      "Number of predicates: {}, number of edges: {}", p, edges.size());

  auto matches = graphutils::findMatchingOps(subgraph, preds, edges);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
}

// Test Op classification with the pipeline model, where:
// - All inputs are through compute tiles
// - All outputs are through compute tiles
BOOST_AUTO_TEST_CASE(DecomposeLoopPipelineClassifyTest) {
  int numStages   = 3;
  int numParallel = 2;

  // Build test model
  ExplicitPipelineTestModel0 testModel(numStages, numParallel, {}, {});

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  // Unroll skewed by pipeline stage such that the main loop ends up
  // overlapping all pipeline stages
  DecomposeTopoConLevel before = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel loop   = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel after  = DecomposeTopoConLevel::Full;

  DecomposeLoopPipelineModel decomposeModel(1,
                                            numStages - 1,
                                            numStages,
                                            before,
                                            loop,
                                            after,
                                            computeLikeExchangeStrategies);

  auto opTypeMap = decomposeModel.classifyOperations(*testModel.subgraphPt);

  for (auto &opType : opTypeMap) {
    logging::trace(
        "Op: {}, type: {}", opType.first->debugName(), opType.second);
  }

  auto &subgraph = *testModel.subgraphPt;

  // Build predicates and edges to check if the graph and classifications are
  // as expected.

  // Number of predicates: We expect 9 ops per path, based on the testModel.
  graphutils::OpPreds preds(9, [](const Op *op) { return false; });
  graphutils::Edges edges;

  int p      = 0;
  preds[p++] = [&opTypeMap](const Op *op) {
    return op->isConvertibleTo<InitOp>() &&
           op->settings.tileSet == TileSet::Compute &&
           opTypeMap.at(op->getGraph().getOp(op->id)) ==
               DecomposeLoopOpPipelineType::auxBeforeComputeLike(0);
  };
  preds[p++] = [&opTypeMap](const Op *op) {
    return op->isConvertibleTo<HostLoadOp>() &&
           op->settings.tileSet == TileSet::Compute &&
           opTypeMap.at(op->getGraph().getOp(op->id)) ==
               DecomposeLoopOpPipelineType::ioBeforeComputeLike(0);
  };
  // Insert an edge between the previous two predicates
  edges.insert({p - 2, p - 1});
  for (int j = 0; j < numStages; ++j) {
    preds[p++] = [j, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<MatMulOp>() &&
             op->settings.tileSet == TileSet::Compute &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 DecomposeLoopOpPipelineType::compute(j);
    };
    edges.insert({p - 2, p - 1});
    if (j < numStages - 1) {
      preds[p++] = [j, &opTypeMap](const Op *op) {
        return op->isConvertibleTo<IpuCopyOp>() &&
               op->settings.tileSet == TileSet::Compute &&
               opTypeMap.at(op->getGraph().getOp(op->id)) ==
                   DecomposeLoopOpPipelineType::computePipelineIpuCopy(j);
      };
      edges.insert({p - 2, p - 1});
    }
  }
  preds[p++] = [numStages, &opTypeMap](const Op *op) {
    return op->isConvertibleTo<AccumulateOp>() &&
           op->settings.tileSet == TileSet::Compute &&
           opTypeMap.at(op->getGraph().getOp(op->id)) ==
               DecomposeLoopOpPipelineType::compute(numStages - 1);
  };
  edges.insert({p - 2, p - 1});
  preds[p++] = [numStages, &opTypeMap](const Op *op) {
    return op->isConvertibleTo<HostStoreOp>() &&
           op->settings.tileSet == TileSet::Compute &&
           opTypeMap.at(op->getGraph().getOp(op->id)) ==
               DecomposeLoopOpPipelineType::ioAfterComputeLike(numStages - 1);
  };
  edges.insert({p - 3, p - 1});
  logging::trace(
      "Number of predicates: {}, number of edges: {}", p, edges.size());

  auto matches = graphutils::findMatchingOps(subgraph, preds, edges);

  // Match two identical paths
  BOOST_REQUIRE_EQUAL(matches.size(), 2);
}

// Test Op classification with the pipeline model, where:
// - All inputs are through IO tiles
//    (1 with ExchangeStrategy::OverlapInnerLoop)
//    (1 with ExchangeStrategy::JustInTime)
// - All outputs are through IO tiles
//    (1 with ExchangeStrategy::OverlapInnerLoop)
//    (1 with ExchangeStrategy::JustInTime)
BOOST_AUTO_TEST_CASE(DecomposeLoopPipelineClassifyTestOverlap) {
  int numStages   = 3;
  int numParallel = 2;

  // Build test model
  ExplicitPipelineTestModel0 testModel(
      numStages,
      numParallel,
      {{0, {TileSet::IO, ExchangeStrategy::OverlapInnerLoop}},
       {1, {TileSet::IO, ExchangeStrategy::JustInTime}}},
      {{0, {"all", TileSet::IO, ExchangeStrategy::OverlapInnerLoop}},
       {1, {"all", TileSet::IO, ExchangeStrategy::JustInTime}}});

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  // Unroll skewed by pipeline stage such that the main loop ends up
  // overlapping all pipeline stages
  DecomposeTopoConLevel before = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel loop   = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel after  = DecomposeTopoConLevel::Full;

  DecomposeLoopPipelineModel decomposeModel(1,
                                            numStages - 1,
                                            numStages,
                                            before,
                                            loop,
                                            after,
                                            computeLikeExchangeStrategies);

  auto opTypeMap = decomposeModel.classifyOperations(*testModel.subgraphPt);

  for (auto &opType : opTypeMap) {
    logging::trace(
        "Op: {}, type: {}", opType.first->debugName(), opType.second);
  }

  auto &subgraph = *testModel.subgraphPt;

  // Build predicates and edges to check if the graph and classifications are
  // as expected.

  // Number of predicates: We expect 11 ops per path, based on the testModel.
  graphutils::OpPreds preds(numParallel * 11,
                            [](const Op *op) { return false; });
  graphutils::Edges edges;

  int p = 0;
  for (int i = 0; i < numParallel; ++i) {
    preds[p++] = [i, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<InitOp>() &&
             op->settings.tileSet == TileSet::IO &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 (i == 0
                      ? DecomposeLoopOpPipelineType::auxBefore(0)
                      : DecomposeLoopOpPipelineType::auxBeforeComputeLike(0));
    };
    preds[p++] = [i, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<HostLoadOp>() &&
             op->settings.tileSet == TileSet::IO &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 (i == 0 ? DecomposeLoopOpPipelineType::ioBefore(0)
                         : DecomposeLoopOpPipelineType::ioBeforeComputeLike(0));
    };
    // Insert an edge between the previous two predicates
    edges.insert({p - 2, p - 1});
    preds[p++] = [i, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<IoTileCopyOp>() &&
             op->settings.tileSet == TileSet::Compute &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 (i == 0 ? DecomposeLoopOpPipelineType::ioToCompute(0)
                         : DecomposeLoopOpPipelineType::compute(0));
    };
    edges.insert({p - 2, p - 1});
    for (int j = 0; j < numStages; ++j) {
      preds[p++] = [j, &opTypeMap](const Op *op) {
        return op->isConvertibleTo<MatMulOp>() &&
               op->settings.tileSet == TileSet::Compute &&
               opTypeMap.at(op->getGraph().getOp(op->id)) ==
                   DecomposeLoopOpPipelineType::compute(j);
      };
      edges.insert({p - 2, p - 1});
      if (j < numStages - 1) {
        preds[p++] = [j, &opTypeMap](const Op *op) {
          return op->isConvertibleTo<IpuCopyOp>() &&
                 op->settings.tileSet == TileSet::Compute &&
                 opTypeMap.at(op->getGraph().getOp(op->id)) ==
                     DecomposeLoopOpPipelineType::computePipelineIpuCopy(j);
        };
        edges.insert({p - 2, p - 1});
      }
    }
    preds[p++] = [numStages, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<AccumulateOp>() &&
             op->settings.tileSet == TileSet::Compute &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 DecomposeLoopOpPipelineType::compute(numStages - 1);
    };
    edges.insert({p - 2, p - 1});
    preds[p++] = [i, numStages, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<IoTileCopyOp>() &&
             op->settings.tileSet == TileSet::IO &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 (i == 0
                      ? DecomposeLoopOpPipelineType::computeToIO(numStages - 1)
                      : DecomposeLoopOpPipelineType::compute(numStages - 1));
    };
    edges.insert({p - 3, p - 1});
    preds[p++] = [i, numStages, &opTypeMap](const Op *op) {
      return op->isConvertibleTo<HostStoreOp>() &&
             op->settings.tileSet == TileSet::IO &&
             opTypeMap.at(op->getGraph().getOp(op->id)) ==
                 (i == 0 ? DecomposeLoopOpPipelineType::ioAfter(numStages - 1)
                         : DecomposeLoopOpPipelineType::ioAfterComputeLike(
                               numStages - 1));
    };
    edges.insert({p - 2, p - 1});
  }
  logging::trace(
      "Number of predicates: {}, number of edges: {}", p, edges.size());

  auto matches = graphutils::findMatchingOps(subgraph, preds, edges);
  BOOST_REQUIRE_EQUAL(matches.size(), 1);
}

// Test loop decomposition with the pipeline model, where:
// - All inputs are through compute tiles
// - All outputs are through compute tiles
BOOST_AUTO_TEST_CASE(DecomposeLoopPipelineModelTest) {
  int numStages   = 3;
  int numParallel = 2;

  // Build test model
  ExplicitPipelineTestModel0 testModel(numStages, numParallel, {}, {});

  auto &graph    = *testModel.graphPt;
  auto &subgraph = *testModel.subgraphPt;

  DecomposeLoops decomposer;

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  // Unroll skewed by pipeline stage such that the main loop ends up
  // overlapping all pipeline stages
  DecomposeTopoConLevel before = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel loop   = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel after  = DecomposeTopoConLevel::Full;

  // Check the trip count before unrolling
  BOOST_REQUIRE_EQUAL(testModel.loopOp->getTripCountValue(), numStages + 2);

  DecomposeLoopPipelineModel decomposeModel(1,
                                            numStages - 1,
                                            numStages,
                                            before,
                                            loop,
                                            after,
                                            computeLikeExchangeStrategies);

  decomposer.decomposeLoop(
      testModel.getIr().getMainGraph(), testModel.loopOp, decomposeModel);

  // Check the decreased trip count after unrolling:
  // We expect 2 fewer iterations:
  // - Before the loop, pipeline stage 0 runs 2 times, stage 1 runs 1 time
  // - All pipeline stages run once within the loop
  // - After the loop, pipeline stage 1 runs 1 time, stage 2 runs 2 times
  BOOST_REQUIRE_EQUAL(testModel.loopOp->getTripCountValue(), numStages);

  // Matches before the LoopOp:
  {
    graphutils::OpPreds preds(11, [](const Op *op) { return false; });
    graphutils::Edges edges;

    int p = 0;

    preds[p++] = [](const Op *op) { return op->isConvertibleTo<LoopOp>(); };

    // Loop over number of unrolls (2)
    for (int k = 0; k < 2; ++k) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<InitOp>(); };
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<HostLoadOp>();
      };
      edges.insert({p - 2, p - 1});
      // Loop over number of stages per unroll
      for (int j = 0; j < k + 1; ++j) {
        preds[p++] = [](const Op *op) {
          return op->isConvertibleTo<MatMulOp>();
        };
        edges.insert({p - 2, p - 1});
        preds[p++] = [](const Op *op) {
          return op->isConvertibleTo<IpuCopyOp>();
        };
        edges.insert({p - 2, p - 1});
        // Due to DecomposeTopoConLevel::Full, we expect a topoCon between
        // compute and IpuCopyOp
        edges.insert({p - 2, p - 1, graphutils::EdgeType::TopoCon});
        if (j == k) {
          // Edge to the LoopOp
          edges.insert({p - 1, 0});
          // Due to DecomposeTopoConLevel::Full, we expect a topoCon between
          // compute and IpuCopyOp
          edges.insert({p - 1, 0, graphutils::EdgeType::TopoCon});
        }
      }
    }

    logging::trace(
        "Number of predicates: {}, number of edges: {}", p, edges.size());

    auto matches = graphutils::findMatchingOps(graph, preds, edges);

    // Since we have two parallel paths (A, B), each with a long (0) and
    // short (1) path to the LoopOp duplicate matches are expected,
    // i.e. it will find {A0, A1}, {A0, B1}, {B0, A1}, {B0, B1}
    BOOST_REQUIRE_EQUAL(matches.size(), 4);
  }

  // Matches in the LoopOp body subgraph:
  {
    graphutils::OpPreds preds(9, [](const Op *op) { return false; });
    graphutils::Edges edges;

    int p = 0;

    preds[p++] = [](const Op *op) { return op->isConvertibleTo<InitOp>(); };
    preds[p++] = [](const Op *op) { return op->isConvertibleTo<HostLoadOp>(); };
    edges.insert({p - 2, p - 1});
    // Loop over number of stages
    for (int j = 0; j < numStages; ++j) {
      preds[p++] = [j](const Op *op) {
        // Ensure the MatMulOp for stage 1 and higher consumes only graph (loop)
        // inputs. The data input is an explicit loop input, while the weight
        // is an implicit loop input
        return op->isConvertibleTo<MatMulOp>() &&
               (j == 0 || op->inTensor(MatMulOp::getLhsInIndex())
                              ->isExplicitLoopInput()) &&
               op->inTensor(MatMulOp::getRhsInIndex())->isImplicitLoopInput();
      };
      if (j == 0) {
        // First stage consumes HostLoadOp input
        edges.insert({p - 2, p - 1});
      }
      if (j < numStages - 1) {
        preds[p++] = [](const Op *op) {
          // Ensure the IpuCopyOp output is a graph output
          return op->isConvertibleTo<IpuCopyOp>() &&
                 op->outTensor(0)->isGraphOutput();
        };
        edges.insert({p - 2, p - 1});
        // Due to DecomposeTopoConLevel::Full, we expect a topoCon between
        // compute and IpuCopyOp
        edges.insert({p - 2, p - 1, graphutils::EdgeType::TopoCon});
      } else {
        preds[p++] = [](const Op *op) {
          return op->isConvertibleTo<AccumulateOp>();
        };
        edges.insert({p - 2, p - 1});
        preds[p++] = [](const Op *op) {
          return op->isConvertibleTo<HostStoreOp>();
        };
        edges.insert({p - 3, p - 1});
      }
    }

    logging::trace(
        "Number of predicates: {}, number of edges: {}", p, edges.size());

    auto matches = graphutils::findMatchingOps(subgraph, preds, edges);

    // Expected matches are 2, but due to permutations of graph structures,
    // there are 8 total matches (2 parallel paths with 3 stages each).
    BOOST_REQUIRE_EQUAL(matches.size(), 8);
  }

  // Matches after the LoopOp:
  {
    graphutils::OpPreds preds(9, [](const Op *op) { return false; });
    graphutils::Edges edges;

    int p = 0;

    preds[p++] = [](const Op *op) { return op->isConvertibleTo<LoopOp>(); };

    // Loop over number of unrolls (2)
    for (int k = 0; k < 2; ++k) {
      // Loop over number of stages per unroll
      for (int j = 0; j < k + 1; ++j) {
        preds[p++] = [](const Op *op) {
          return op->isConvertibleTo<MatMulOp>();
        };
        if (j == 0) {
          // From LoopOp output
          edges.insert({0, p - 1});
        } else {
          // From IpuCopyOp
          edges.insert({p - 2, p - 1});
        }
        if (j < k) {
          preds[p++] = [](const Op *op) {
            return op->isConvertibleTo<IpuCopyOp>();
          };
          edges.insert({p - 2, p - 1});
          // Due to DecomposeTopoConLevel::Full, we expect a topoCon between
          // compute and IpuCopyOp
          edges.insert({p - 2, p - 1, graphutils::EdgeType::TopoCon});
        }
      }
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<AccumulateOp>();
      };
      edges.insert({p - 2, p - 1});
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<HostStoreOp>();
      };
      edges.insert({p - 3, p - 1});
    }

    logging::trace(
        "Number of predicates: {}, number of edges: {}", p, edges.size());

    auto matches = graphutils::findMatchingOps(graph, preds, edges);

    // Since we have two parallel paths (A, B), each with a long (0) and
    // short (1) path to the LoopOp duplicate matches are expected,
    // i.e. it will find {A0, A1}, {A0, B1}, {B0, A1}, {B0, B1}
    BOOST_REQUIRE_EQUAL(matches.size(), 4);
  }

  // Check if the LoopOp promotes aliases and modifies correctly
  int numModifies = 0;
  int numAliased  = 0;
  for (auto input : testModel.loopOp->input->tensorMap()) {
    auto subgraphInIndex = testModel.loopOp->opInToSubgraphInIndex(input.first);
    for (auto c : testModel.loopOp->getCalledGraph()
                      .getInputTensor(subgraphInIndex)
                      ->consumers.getOps()) {
      // Check that the whole accumulator is modified by the LoopOp
      if (c->isConvertibleTo<AccumulateOp>()) {
        if (!testModel.loopOp->modifies(input.first).empty() &&
            testModel.loopOp->modifies(input.first).front() ==
                view::Region::getFull({4, 4}, view::AccessType::ReadWrite)) {
          ++numModifies;
        }
        auto sgOutTensor = c->outTensor(AccumulateOp::getUpdatedVarOutIndex());
        if (sgOutTensor->isGraphOutput()) {
          auto aliasedRegions = testModel.loopOp->aliases(
              input.first,
              testModel.loopOp->subgraphOutToOpOutIndex(
                  sgOutTensor->getGraphOutputIndex()));
          if (!aliasedRegions.empty() &&
              aliasedRegions.front() ==
                  view::Region::getFull({4, 4}, view::AccessType::ReadWrite)) {
            ++numAliased;
          }
        }
      }
    }
  }
  // Two modified accumulators
  BOOST_REQUIRE_EQUAL(numModifies, 2);
  // Two aliased accumulators
  BOOST_REQUIRE_EQUAL(numAliased, 2);
}

// Test loop decomposition with the pipeline model, where:
// - All inputs are through IO tiles
//    (1 with ExchangeStrategy::OverlapInnerLoop)
//    (1 with ExchangeStrategy::JustInTime)
// - All outputs are through IO tiles
//    (1 with ExchangeStrategy::OverlapInnerLoop)
//    (1 with ExchangeStrategy::JustInTime)
BOOST_AUTO_TEST_CASE(DecomposeLoopPipelineModelTestOverlap) {
  int numStages   = 3;
  int numParallel = 2;

  // Build test model
  ExplicitPipelineTestModel0 testModel(
      numStages,
      numParallel,
      {{0, {TileSet::IO, ExchangeStrategy::OverlapInnerLoop}},
       {1, {TileSet::IO, ExchangeStrategy::JustInTime}}},
      {{0, {"all", TileSet::IO, ExchangeStrategy::OverlapInnerLoop}},
       {1, {"all", TileSet::IO, ExchangeStrategy::JustInTime}}});

  auto &subgraph = *testModel.subgraphPt;

  DecomposeLoops decomposer;

  std::set<ExchangeStrategy> computeLikeExchangeStrategies = {
      ExchangeStrategy::JustInTime};

  // Unroll skewed by pipeline stage such that the main loop ends up
  // overlapping all pipeline stages
  DecomposeTopoConLevel before = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel loop   = DecomposeTopoConLevel::Full;
  DecomposeTopoConLevel after  = DecomposeTopoConLevel::Full;

  DecomposeLoopPipelineModel decomposeModel(0,
                                            numStages,
                                            numStages,
                                            before,
                                            loop,
                                            after,
                                            computeLikeExchangeStrategies);

  // Check the trip count before unrolling
  BOOST_REQUIRE_EQUAL(testModel.loopOp->getTripCountValue(), numStages + 2);

  decomposer.decomposeLoop(
      testModel.getIr().getMainGraph(), testModel.loopOp, decomposeModel);

  // Check the decreased trip count after unrolling
  // We expect 4 fewer iterations:
  // - The two extra unroll steps facilitate overlapped IO
  // - Before the loop, pipeline stage 0 runs 3 times, stage 1 runs 2 times,
  // stage 2 runs 1 time
  // - All pipeline stages run once within the loop
  // - After the loop, pipeline stage 0 runs 1 time, stage 1 runs 2 times, stage
  // 2 runs 3 times
  BOOST_REQUIRE_EQUAL(testModel.loopOp->getTripCountValue(), numStages - 2);

  // Matches in the LoopOp body subgraph:
  {
    graphutils::OpPreds preds(numParallel * 11,
                              [](const Op *op) { return false; });
    graphutils::Edges edges;

    int p = 0;

    // Loop over number of parallel paths
    for (int i = 0; i < numParallel; ++i) {
      preds[p++] = [](const Op *op) { return op->isConvertibleTo<InitOp>(); };
      preds[p++] = [](const Op *op) {
        return op->isConvertibleTo<HostLoadOp>();
      };
      edges.insert({p - 2, p - 1});
      preds[p++] = [i](const Op *op) {
        // With overlapped IO, IoTileCopyOp output is a graph output
        return op->isConvertibleTo<IoTileCopyOp>() &&
               (i == 1 ||
                op->outTensor(IoTileCopyOp::getOutIndex())->isGraphOutput());
      };
      edges.insert({p - 2, p - 1});
      // Loop over number of stages
      for (int j = 0; j < numStages; ++j) {
        preds[p++] = [i, j](const Op *op) {
          // Ensure the MatMulOp for stage 1 and higher consumes only graph
          // (loop) inputs. For path i == 0, with overlapped IO, ensure all
          // stages only consume graph inputs.
          // The data input is an explicit loop input, while the
          // weight is an implicit loop input
          return op->isConvertibleTo<MatMulOp>() &&
                 ((i == 1 && j == 0) || op->inTensor(MatMulOp::getLhsInIndex())
                                            ->isExplicitLoopInput()) &&
                 op->inTensor(MatMulOp::getRhsInIndex())->isImplicitLoopInput();
        };
        if (j == 0) {
          if (i == 1) {
            // First stage consumes HostLoadOp input, if not overlapping
            edges.insert({p - 2, p - 1});
          }
        }
        if (j < numStages - 1) {
          preds[p++] = [](const Op *op) {
            // Ensure the IpuCopyOp output is a graph output
            return op->isConvertibleTo<IpuCopyOp>() &&
                   op->outTensor(0)->isGraphOutput();
          };
          edges.insert({p - 2, p - 1});
          // Due to DecomposeTopoConLevel::Full, we expect a topoCon between
          // compute and IpuCopyOp
          edges.insert({p - 2, p - 1, graphutils::EdgeType::TopoCon});
        } else {
          preds[p++] = [](const Op *op) {
            return op->isConvertibleTo<AccumulateOp>();
          };
          edges.insert({p - 2, p - 1});
          preds[p++] = [i](const Op *op) {
            // With overlapped IO, IoTileCopyOp output is a graph output, and
            // the HostStore is delayed to the next iteration
            return op->isConvertibleTo<IoTileCopyOp>() &&
                   (i == 1 || op->outTensor(IoTileCopyOp::getOutIndex())
                                  ->isGraphOutput());
          };
          edges.insert({p - 3, p - 1});
          preds[p++] = [i](const Op *op) {
            // With overlapped IO, HostStoreOp input is a graph input
            return op->isConvertibleTo<HostStoreOp>() &&
                   (i == 1 || op->inTensor(HostStoreOp::getLocalTensorInIndex())
                                  ->isExplicitLoopInput());
          };
          if (i == 1) {
            // Last stage stores output, if not overlapping
            edges.insert({p - 2, p - 1});
          }
        }
      }
    }

    logging::trace(
        "Number of predicates: {}, number of edges: {}", p, edges.size());

    auto matches = graphutils::findMatchingOps(subgraph, preds, edges);

    // Expected matches are 1, but due to permutations of graph structures,
    // there are 6 total matches (which intermediate pipeline stage belongs
    // to which parallel path is ambiguous).
    BOOST_REQUIRE_EQUAL(matches.size(), 6);
  }

  // Check if the LoopOp promotes aliases and modifies correctly
  int numModifies = 0;
  int numAliased  = 0;
  for (auto input : testModel.loopOp->input->tensorMap()) {
    auto subgraphInIndex = testModel.loopOp->opInToSubgraphInIndex(input.first);
    for (auto c : testModel.loopOp->getCalledGraph()
                      .getInputTensor(subgraphInIndex)
                      ->consumers.getOps()) {
      // Check that the whole accumulator is modified by the LoopOp
      if (c->isConvertibleTo<AccumulateOp>()) {
        if (!testModel.loopOp->modifies(input.first).empty() &&
            testModel.loopOp->modifies(input.first).front() ==
                view::Region::getFull({4, 4}, view::AccessType::ReadWrite)) {
          ++numModifies;
        }
        auto sgOutTensor = c->outTensor(AccumulateOp::getUpdatedVarOutIndex());
        if (sgOutTensor->isGraphOutput()) {
          auto aliasedRegions = testModel.loopOp->aliases(
              input.first,
              testModel.loopOp->subgraphOutToOpOutIndex(
                  sgOutTensor->getGraphOutputIndex()));
          if (!aliasedRegions.empty() &&
              aliasedRegions.front() ==
                  view::Region::getFull({4, 4}, view::AccessType::ReadWrite)) {
            ++numAliased;
          }
        }
      }
    }
  }
  // Two modified accumulators
  BOOST_REQUIRE_EQUAL(numModifies, 2);
  // Two aliased accumulators
  BOOST_REQUIRE_EQUAL(numAliased, 2);
}
