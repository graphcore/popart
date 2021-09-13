// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAPH_TEST_MODELS_HPP
#define GUARD_NEURALNET_GRAPH_TEST_MODELS_HPP

#include <popart/dataflow.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/sessionoptions.hpp>

class GraphTestModel {
public:
  GraphTestModel();

  popart::Ir &getIr() { return ir; }

protected:
  popart::DataFlow df;
  popart::Ir ir;
};

class GraphTestModel1 : public GraphTestModel {
public:
  GraphTestModel1();
};

class GraphTestModel2 : public GraphTestModel {
public:
  GraphTestModel2();
};

class GraphTestModel3 : public GraphTestModel {
public:
  /**
   * Construct a test graph with 2 inputs and 1 output
   * The streams for the inputs/outputs are A (in), B (in) and C (out)
   * \param strategyA ExchangeStrategy to set for input 1
   * \param strategyB ExchangeStrategy to set for input 2
   * \param strategyC ExchangeStrategy to set for output 1
   */
  GraphTestModel3(popart::ExchangeStrategy strategyA,
                  popart::ExchangeStrategy strategyB,
                  popart::ExchangeStrategy strategyC);
};

enum class TestOptimizer {
  SGD0 = 0,
  SGD1,
  SGD2,
  Adam,
  Lamb,
  N,
};

class OptimizerTestModel : public GraphTestModel {
public:
  /**
   * Construct simple test graph updating two weights with an optimizer
   * \param TestOptimizer The optimizer to insert into the graph
   * \param accumulationFactor Gradient accumulation factor
   * \param options Options for the session
   */
  OptimizerTestModel(TestOptimizer opt,
                     unsigned accumulationFactor,
                     popart::SessionOptions options);
};

#endif
