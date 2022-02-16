// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAPH_TEST_MODELS_HPP
#define GUARD_NEURALNET_GRAPH_TEST_MODELS_HPP

#include <popart/dataflow.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/replicatedstreammode.hpp>
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
  GraphTestModel4(popart::ReplicatedStreamMode xMode);
};

/**
 * Forwards pass:
 *
 *   [inputs]         [labels]        ["weights"]
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
 *   L1Grad      .-------------------'         |
 *     |         |                             |
 *     |         v                             |
 *     | Transpose                             |
 *     |  |                                    |
 *     v  v                                    |
 *    MatMul                                   |
 *     |                                       |
 * ....(.. [sg1] ......................        |
 * :   | subgraph'ed if SG1::Yes      :        |
 * :   v                              :        |
 * :  ReplicatedAllReduce             :        |
 * :   |                              :        |
 * :...(..............................:        |
 *     |                                       |
 *     |         .-----------------------------'
 *     |         |
 * ....(. [sg2] .(.....................
 * :   |         |                    :
 * :   v         v                    :
 * :  SGD0VarUpdate                   :
 * :   |                              :
 * :   | subgraph'ed if SG2::Yes      :
 * :...(..............................:
 *     |
 *   [...]
 *
 **/
class GraphTestModel5 : public GraphTestModel {
public:
  enum class SG1 {
    No  = 0,
    Yes = 1,
    N,
  };
  enum class SG2 {
    No  = 0,
    Yes = 1,
    N,
  };

  GraphTestModel5(SG1 sg1, SG2 sg2);
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

/**
 * Repeated 4 times with varying shapes and remote buffer IDs:
 *
 *  (InitOp)              (InitOp)
 *      |                     |
 * [D*_sharded] [index_*] [D*_full]
 *      |        |            |
 *     (RemoteLoad)       (ReduceScatter)
 *      |                     |
 * [D*_scattered]         [D*_loaded]
 *      |       .-------------'
 * (CopyVarUpdate)
 *      |
 * [D*_updated]
 *
 * All operators are flagged as optimizer Ops and support RTS.
 *
 */
class RemoteRTSTestModel : public GraphTestModel {
public:
  /**
   * Construct simple test graph with a few disjunct RTS domains using remote
   * variables
   * \param options Options for the session
   */
  RemoteRTSTestModel(popart::SessionOptions options);

  std::vector<popart::Tensor *> domainTensors;

  std::vector<popart::Op *> initOps;
  std::vector<popart::Op *> loadOps;
  std::vector<popart::Op *> reduceScatterOps;
  std::vector<popart::Op *> varUpdateOps;
};

/*
 * Psuedo code:
 *
 *     def subgraph(st0, st1):
 *       return op(st0, st1)
 *
 *     t0 = var()
 *     t1 = var()
 *     t2 = var()
 *     t3 = var()
 *
 *     call(subgraph, t0, t1)
 *     call(subgraph, t2, t3)
 */
class TraverseCallSiteTestModel : public GraphTestModel {
public:
  TraverseCallSiteTestModel();
};

#endif
