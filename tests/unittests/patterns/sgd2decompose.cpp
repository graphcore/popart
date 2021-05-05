// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TestPatternsSGD2Decompose
#include <boost/test/unit_test.hpp>

#include <popart/patterns/sgd2decompose.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/sgd1combo.hpp>
#include <popart/op/sgd2combo.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/patterns/patterns.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestMatchesSGD2ComboOpOnly) {
  Ir ir;
  Graph &graph = ir.getMainGraph();

  // Make any SGD2.
  const auto sgd2 =
      graph.createOp<SGD2ComboOp>(OptimizerValue{},
                                  OptimizerValue{},
                                  OptimizerValue{},
                                  OptimizerValue{},
                                  true,
                                  OptimizerReductionType::AccumReduce,
                                  DataType::FLOAT,
                                  DataType::FLOAT,
                                  Op::Settings(graph, "sgd2"));

  // Make any other op.
  const auto sgd1 =
      graph.createOp<SGD1ComboOp>(OptimizerValue{},
                                  OptimizerValue{},
                                  OptimizerValue{},
                                  OptimizerValue{},
                                  OptimizerReductionType::AccumReduce,
                                  Op::Settings(graph, "sgd1"));

  // Test only matches SGD2.
  SGD2Decompose pat;
  BOOST_REQUIRE(pat.matches(sgd2));
  BOOST_REQUIRE(!pat.matches(sgd1));
}

BOOST_AUTO_TEST_CASE(TestPatternNamesContainsSGD2Decompose) {
  BOOST_REQUIRE_NO_THROW(PatternNames::getName<SGD2Decompose>());
}
