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

/**
 * Populate an IR with the graph from the `basic_graph.cpp` example:
 *
 * Psuedo code:
 *
 *     x = h2d_stream(...)
 *     w = var(0, ...)
 *     w += x
 *     c = const(5)
 *     y = w + c
 *     d2h_stream(y)
 *
 * ASCII representation:
 *
 *               Init
 *                |
 *              [x__t0]
 *                |
 *                v
 *            HostLoad
 *      w         |
 *      |    [x__t0__t1]
 *      |         |
 *      '-----.   |
 *            v   v
 *          Accumulate
 *              |
 *           [w__t2]   [c=5.0f]
 *              | .-------'
 *              | |
 *              v v
 *              Add
 *               |
 *              [y]
 *               |
 *               v
 *           HostStore
 **/
class GraphTestModel4 : public GraphTestModel {
public:
  GraphTestModel4();
  GraphTestModel4(popart::Tensor::ReplicatedStreamMode xMode);
};

/**
 * Forwards pass:
 *
 *   [inputs]         [labels]        [weights  ]
 *     |                 |             |       |
 *     v                 |             |       |
 *   Identity            |             |       |
 *     |                 |             |       |
 *     |    .------------(-------------'       |
 *     |    |            |                     |
 *     |----(------------(-----------.         |
 *     |    |            |           |         |
 *     v    v            |           |         |
 *     MatMul            |           |         |
 *     |                 |           |         |
 *     | .---------------'           |         |
 *     | |                           |         |
 *     v v                           |         |
 *     Sub                           |         |
 *     |                             |         |
 *     |---------------------.       |         |
 *     v                     |       |         |
 *     L1                    |       |         |
 *     |                     |       |         |
 *     v                     |       |         |
 *   [loss:L1:0]             |       |         |
 *                           |       |         |
 * Backwards pass:           |       |         |
 *                           |       |         |
 *    [1]                    |       |         |
 *     |  .------------------'       |         |
 *     v  v                          |         |
 *   L1Grad            .-------------'         |
 *     |               |                       |
 *     |               v                       |
 *     | TransposeInplace                      |
 *     |  |                                    |
 *     v  v                                    |
 *    MatMul                                   |
 *     |                                       |
 *     v                                       |
 *    ReplicatedAllReduce                      |
 *     |                                       |
 *     |         .-----------------------------'
 *     v         v
 *    SGD0VarUpdate
 *     |
 *   [...]
 *
 **/
class GraphTestModel5 : public GraphTestModel {
public:
  GraphTestModel5();
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
