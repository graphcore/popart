// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAPH_TEST_MODELS_HPP
#define GUARD_NEURALNET_GRAPH_TEST_MODELS_HPP

#include <map>
#include <vector>
#include <popart/dataflow.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/replicatedstreammode.hpp>

namespace popart {
class Graph;
class LoopOp;
class Op;
class Tensor;
struct SessionOptions;
} // namespace popart

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
 * IR which contains an autodiffable subgraph "A" as follows (backwards graph
 * not included):
 *
 *                             ............[typical autodiff]...............
 *                             :                                           :
 *    [A/in0]                  :             [_k/getGradId(in0)]           :
 *      | [A/in1]              :               ^ [_k/getGradId(in1)]       :
 *      |  | [A/in2]           :    [_k/in1]   |   ^ [_k/getGradId(in2)]   :
 *    #0|  |#1   |#2           :      |        |   |   ^                   :
 * .----|--|-----|----["A"]-.  : .----|--------|---|---|-----------[_k]--. :
 * |    |  |     |          |  : |    |        '-. |   |                 | :
 * |    |  |     |          |  : |    |          | |   |                 | :
 * |    |  |#1   |          |  : |    |          S S   S  (S=SumOp)      | :
 * |    |  |  .--'          |  : |    |          | |   |                 | :
 * | #0 v  v  v #2          |  : |    |       #0 | |#1 | #2              | :
 * |   [GTM6Op ]            |  : |    |         GTM6GradOp               | :
 * | #0 |  |  | #2          |  : |    |       #0 ^ ^   ^                 | :
 * |    |  |  '-.           |  : |    '----------' |   |                 | :
 * |    |  |#1  |           |  : |              #1 |   | #2              | :
 * '----|--|----|-----------'  : '-----------------|---|-----------------' :
 *    #0|  |#1  v#2            :                   |   |                   :
 *      |  v [A/out2]          :                   | [_k/getGradId(in2)]   :
 *      v [A/out1]             :                 [_k/getGradId(in0)]       :
 *    [A/out0]                 :                                           :
 *                             :...........................................:
 *
 * Note:
 * - While the order of inputs/outputs to GMT6GradOp is fixed, autodiff
 *   decides the order of the backwards graph inputs/outputs, depending on
 *   parameters passed to it.
 * - The GMT6GradOp does not require A/out1 to produce gradients for it's
 *   inputs. In terms of activations, it only needs A/in1.
 **/
class GraphTestModel6 : public GraphTestModel {
public:
  GraphTestModel6();

private:
  class GTM6Op;
  class GTM6GradOp;
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

/**
 * Describes a simple model before the application of explicit recomputation.
 * The model contains skip-connections between pipeline stages to trigger
 * multiple recomputation.
 *
 * Pseudocode of the core model:
 *
 * output = ...
 * for i in range(numLayers):
 *   for j in range(numMatMulsPerLayer):
 *     output = matmul(w_i_j, output);
 *     if i >= 2:
 *       output = add(output, output_of_(i-2)_j)
 *
 * The first operation of each layer is set to `checkpoint`, while all other
 * operations are set to `recompute`. Note that isn't a requirement for either
 * normal or pipelined explicit recompute, but rather to test if user-defined
 * recomputes work in both cases. For pipelining, by default, the IpuCopyOps
 * inserted later will always be checkpointed regardless.
 */
class ExplicitRecomputeTestModel : public GraphTestModel {
public:
  ExplicitRecomputeTestModel(bool pipelining,
                             int numLayers,
                             int numMatMulsPerLayer);
};

/**
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

/**
 * Describes a multi-stage pipeline model, as it would occur before
 * decomposing the loop in the explicit pipeline transform.
 * Contains HostLoadOp, HostStoreOp, IpuCopyOp, AccumulateOp, MatMulOp,
 * IoTileCopyOp.
 *
 * p0,..., pn: numPipelineStages
 *
 *                                     (A0 and A1 are two accumulators that
 *                                      will be passed through and updated
 *                                      inside of the loop by AccumulateOp)
 *
 *                                        A0           A1
 * LoopOp(numPipelineStages + 2):         |            |
 *                                        |            |
 *         (numParallelPaths times)       |            |
 * p0      HostLoad ...  HostLoad         |            |
 *         |             |                |            |
 *        (IoTileCopy)   (IoTileCopy)     |            |
 *         |             |                |            |
 * p1      MatMul        MatMul           |            |
 *         |             |                |            |
 *         IpuCopy       IpuCopy          |            |
 *         |             |                |            |
 *         MatMul        MatMul           |            |
 *         ...           ...              |            |
 *         |             |                |            |
 * pn      MatMul        Matmul ----------|------> Accumulate
 *         |     \-------|----------> Accumulate       |
 *        (IoTileCopy)   (IoTileCopy)     |            |
 *         |             |                |            |
 *         HostStore     HostStore        |            |
 *                                        A0'          A1'
 *
 *                                (A0', A1': Updated, aliased accumulator state)
 */
class ExplicitPipelineTestModel0 : public GraphTestModel {
public:
  /**
   * Construct the ExplicitPipelineTestModel
   * \param numPipelineStages  Number of pipeline stages
   * \param numParallelPaths   Number of paths through each pipeline stage
   * \param inputExStrategy    TileSet and ExchangeStrategy for each input of
   *                           each path
   * \param outputExStrategy   TileSet and ExchangeStrategy for each output of
   *                           each path
   */
  ExplicitPipelineTestModel0(
      int numPipelineStages,
      int numParallelPaths,
      std::map<int, popart::InputSettings> inputExStrategy,
      std::map<int, popart::AnchorReturnType> outputExStrategy);

  popart::LoopOp *loopOp;
  popart::Graph *graphPt;
  popart::Graph *subgraphPt;
};

/**
 * Describes a simple model before the application of explicit pipelining.
 * The model contains skip-connections between pipeline stages to trigger
 * multiple restores.
 *
 * Pseudocode of the core model:
 *
 * output = ...
 * for i in range(numLayers):
 *   for j in range(numMatMulsPerLayer):
 *     output = matmul(w_i_j, output);
 *     if i >= 2:
 *       output = add(output, output_of_(i-2)_j)
 *
 * Transforms, which are not under-test, applied to the model:
 * - constructBackwards
 * - InterIpuCopy
 * - MainLoops
 *
 * (in order to avoid construction by hand)
 */
class ExplicitPipelineTestModel1 : public GraphTestModel {
public:
  ExplicitPipelineTestModel1(int numLayers, int numMatMulsPerLayer);

  popart::LoopOp *loopOp;
  popart::Graph *graphPt;
  popart::Graph *subgraphPt;
};

#endif
