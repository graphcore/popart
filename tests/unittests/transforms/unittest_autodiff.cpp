// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE autodiff_unittest

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/sum.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensornames.hpp>

#define private public
#include <popart/transforms/autodiff.hpp>
#undef private

#include <testutil/irquery/irquery.hpp>

using namespace popart;
using namespace popart::irquery;

// In tests below we use TestOps to build up a popart::Ir object. We apply
// the autodiff transform on this Ir and check that the resulting Ir is
// what we expect.

namespace {

// Forward declarations.
class TestOp;
class SimpleTestOp;
class SimpleTestGradOp;

/**
 * Base class for TestOps in this file.
 **/
class TestOp : public Op {
public:
  TestOp(const OperatorIdentifier &opid, const Op::Settings &settings)
      : Op(opid, settings) {}

  virtual float getSubgraphValue() const override {
    return getLowSubgraphValue();
  }
};

/**
 * An op with the following behaviour:
 *  - an input #0: (details unimportant)
 *  - an output #0: (same tensor info as input)
 *  - calls to getGradOps() return {SimpleTestGradOp}
 *  - otherwise default popart::Op behaviour
 **/
class SimpleTestOp : public TestOp {
public:
  SimpleTestOp(const Op::Settings &settings)
      : TestOp(OperatorIdentifier("TestOps", "SimpleTestOp", 1), settings) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<SimpleTestOp>(*this);
  }

  // Defined below because SimpleTestGradOp is not defined yet.
  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> grads;
    grads.emplace_back(std::make_unique<SimpleTestGradOp>(*this));
    return grads;
  }
};

/**
 * An op with the following behaviour:
 *  - an input #0: the gradient of a SimpleTestOp's #0 output
 *  - an input #1: the #0 input of a SimpleTestOp
 *  - an output #0: the gradient of a SimpleTestOp's #0 input=
 *  - otherwise default popart::Op behaviour
 **/
class SimpleTestGradOp : public TestOp {
public:
  SimpleTestGradOp(const SimpleTestOp &op)
      : TestOp(OperatorIdentifier("TestOps", "SimpleTestGradOp", 1),
               op.Op::getSettings()) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<SimpleTestGradOp>(*this);
  }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {0, 0, GradOpInType::GradOut}, {1, 0, GradOpInType::In}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }
};

/**
 * Helper function to print the IR.
 */
void printIr(Ir &ir) {
  std::stringstream ss;
  ir.append(ss);
  std::cout << ss.str() << std::endl;
}

/**
 * Receive a mapping from indices to expected connections and output a list
 * of expected connections.
 **/
ExpectedConnections
sortExpectedConnections(const std::map<int, ExpectedConnection> &map) {
  ExpectedConnections result;
  result.resize(map.size());
  for (const auto &kv : map) {
    result.at(kv.first) = kv.second;
  }
  return result;
}

/**
 * Helper function that gets the unique index of an input.
 */
InIndex getUniqInIndex(Op *op, const TensorId &id) {
  const auto &indices = op->input->indices(op->getGraph().getTensors().get(id));
  BOOST_REQUIRE(indices.size() == 1);
  return indices.at(0);
};

} // namespace

BOOST_AUTO_TEST_CASE(autodiff_0) {

  // Check that autodiff (`bool apply(Ir&)`) adds parts of the IR marked below.
  //
  // This is a very basic IR that comprises of only a main graph with one simple
  // test op.
  //
  // .----------------[main graph]-------------------.
  // |                                               |
  // |                ....[added by autodiff]....... |
  // |                :                            : |
  // |    ["t0"]      :        [getGradId("t0")]   : |
  // |      |         :                  ^         : |
  // |      |         :                  |         : |
  // |      |         :                 Sum        : |
  // |      |         :                  ^         : |
  // |      |---------(-----.            |         : |
  // |      |         :     |           [_g0]      : |
  // |      |         :     |            |         : |
  // |   #0 v         :     |         #0 |         : |
  // | SimpleTestOp   :     | SimpleTestGradOp     : |
  // |   #0 |         :     | #1 ^    #0 ^         : |
  // |      |         :     |    |       |         : |
  // |      |         :     '----'       |         : |
  // |      v         :                  |         : |
  // |    ["t1"]      :      [getGradId("t1")]     : |
  // |                :                            : |
  // |                :                            : |
  // |                :............................: |
  // '-----------------------------------------------'
  //
  // Where t1 is marked as the loss and t0 is a variable tensor and
  //
  // - _g0    is an arbitrary edge gradient TensorId.

  Ir ir;

  // First we build up the Ir's left hand side as per the diagram above. Note
  // that we avoid using the builder to do this (which is used in most tests)
  // to try and minimise the amount of production code we are instantiating in
  // this test, making it as close to a unit test as possible.
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto simpleTestOp = mainGraph.createOp<SimpleTestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  simpleTestOp->connectInTensor(0, "t0");
  simpleTestOp->createAndConnectOutTensor(0, "t1");
  simpleTestOp->setup();

  // Set "t1" as the loss.
  ir.setFinalLoss("t1");
  ir.getMainGraph().setLoss("t1");

  // NOTE: We need to do a few boilerplate things to apply autodiff. In future,
  // we'd like to refactor autodiff so this isn't the case. We also need an
  // optimizer for autodiff to work.
  ir.setOptimizer(ConstSGD(0.1));
  ir.updateVertices();

  // Now apply the transform.
  ir.applyTransform(Autodiff::id(), mainGraph);

  // Now check that autodiff added the right things.

  // Start by getting a test wrapper for the main graph.
  IrTestWrapper tw_ir{ir};
  auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);

  // Now find the SimpleTestGradOp that consumed "t0" and
  // getGradId("t1") and produces _g0.
  auto tw_simpleTestGradOp = tw_mainGraph->ops().hasOp<SimpleTestGradOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasIdAtIndex(0, getGradId("t1")) &&
                (tw_op.inputs().hasIdAtIndex(1, "t0")) &&
                (tw_op.outputs().hasIndex(0)));
      },
      Require::MustBeTrue);

  // Now, get the value of _g0.
  auto _g0 =
      tw_simpleTestGradOp->outputs().hasIndex(0, Require::MustBeTrue)->id();

  // Find the gradsum op that consumes _g0 and produces getGradId("t0").
  auto tw_gradSumOp = tw_mainGraph->ops().hasOp<SumOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasExactIds({_g0})) &&
               (tw_op.outputs().hasIdAtIndex(0, getGradId("t0")));
      },
      Require::MustBeTrue);
}

BOOST_AUTO_TEST_CASE(autodiff_1) {

  // Check that autodiff (`bool apply(Ir&)`) adds parts of the IR marked below.
  //
  // This test case comprises a subgraph with two test ops and a main graph
  // with one callop to the subgraph.
  //
  // .-----------------------------[main graph]--------------------------------.
  // |                                                                         |
  // |                           ............[added by autodiff].............. |
  // |                           :                                           : |
  // |  ["main_in"]              :       [getGradId("main_in")]              : |
  // |      |                    :                    ^                      : |
  // |      |                    :                    |                      : |
  // |      |                    :                   Sum                     : |
  // |      |                    :                    ^                      : |
  // |      |--------------------(-------.            |                      : |
  // |      |                    :       |          [_g0]                    : |
  // |      |                    :       |            |                      : |
  // |   #0 v                    :       |         #0 |                      : |
  // |  CallOp<A>                :       |     CallOp<_k>                    : |
  // |   #0 |                    :       |  #_x ^ #_y ^                      : |
  // |      |                    :       |      |     |                      : |
  // |      |                    :       '------'     |                      : |
  // |      v                    :                    |                      : |
  // |  ["main_out"]             :      [getGradId("main_out")]              : |
  // |                           :                                           : |
  // |                           :                                           : |
  // |                           :...........................................: |
  // '-------------------------------------------------------------------------'
  //
  //                             ............[added by autodiff]..............
  //                             :                                           :
  //    [A/a_in]                 :    [_k/a_in]         [_k/getGradId(a_in)] :
  //      |                      :  #_x |                       #0 ^         :
  // .----|-[subgraph "A"]----.  : .----|--------[subgraph _k]-----|-------. :
  // |    |                   |  : |    |----------.              Sum      | :
  // |    |                   |  : |    |          |               ^       | :
  // |    |                   |  : |    |          |               |       | :
  // |    |                   |  : |    |          |             [_g2]     | :
  // | #0 v                   |  : | #0 v          |            #0 |       | :
  // | SimpleTestOp           |  : | SimpleTestOp  |    SimpleTestGradOp   | :
  // | #0 |                   |  : | #0 |          |    #1 ^    #0 ^       | :
  // |    |                   |  : |    |          |       |       |       | :
  // |    |                   |  : |    |          '-------'       |       | :
  // |  [A/a_tmp]             |  : | [_k/a_tmp]                    |       | :
  // |    |                   |  : |    |          [_k/getGradId(a_tmp)]   | :
  // |    |                   |  : |    |                          ^       | :
  // |    |                   |  : |    |                          |       | :
  // |    |                   |  : |    |                         Sum      | :
  // |    |                   |  : |    |                          ^       | :
  // |    |                   |  : |    |                          |       | :
  // |    |                   |  : |    |                        [_g1]     | :
  // | #0 v                   |  : |    |                       #0 |       | :
  // | SimpleTestOp           |  : |    |               SimpleTestGradOp   | :
  // | #0 |                   |  : |    |               #1 ^    #0 ^       | :
  // |    |                   |  : |    |                  |       |       | :
  // |    |                   |  : |    '------------------'       |       | :
  // '----|-------------------'  : '-------------------------------|-------' :
  //      v                      :                             #_y |         :
  //    [A/a_out]                :                     [_k/getGradId(a_out)] :
  //                             :...........................................:
  //
  // Where:
  // - _k              is an arbitrary GraphId.
  // - _x, _y          are arbitrary InIndices.
  // - _g0, _g1, _g2   are arbitrary edge gradient TensorIds.

  Ir ir;

  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  int32_t tData[] = {5};

  // Create the subgraph.
  auto &mainGraph = ir.getMainGraph();
  auto &subgraphA = ir.createGraph(GraphId("A"));

  // Create subgraph A.
  auto a_in  = subgraphA.addScope("a_in");
  auto a_tmp = subgraphA.addScope("a_tmp");
  auto a_out = subgraphA.addScope("a_out");
  subgraphA.addInput(a_in, tInfo);

  // Add SimpleTestOp0.
  Op::Settings settingsSubgraphA = Op::Settings{subgraphA, "SimpleTestOp"};
  auto simpleTestOp0 = subgraphA.createOp<SimpleTestOp>(settingsSubgraphA);

  // Connect SimpleTestOp0.
  simpleTestOp0->connectInTensor(0, a_in);
  simpleTestOp0->createAndConnectOutTensor(0, a_tmp);
  simpleTestOp0->setup();

  // Add SimpleTestOp1.
  auto simpleTestOp1 = subgraphA.createOp<SimpleTestOp>(settingsSubgraphA);

  // Connect SimpleTestOp1.
  simpleTestOp1->connectInTensor(0, a_tmp);
  simpleTestOp1->createAndConnectOutTensor(0, a_out);
  simpleTestOp1->setup();

  // Mark "a_out" as a graph output.
  subgraphA.markAsOutput(a_out);

  // Create main graph.
  mainGraph.getTensors().addVarInit(
      "main_in", tInfo, static_cast<void *>(&tData));

  auto callOp = mainGraph.createOp<CallOp>(
      Onnx::AiGraphcore::OpSet1::Call,
      subgraphA,
      Op::Settings{mainGraph, "", mainGraph.getScope()});

  callOp->connectInTensor(0, "main_in");
  callOp->createAndConnectOutTensor(0, "main_out");
  callOp->setup();

  // Set "t1" as the loss.
  ir.setFinalLoss("main_out");
  ir.getMainGraph().setLoss("main_out");

  // NOTE: We need to do a few boilerplate things to apply autodiff. In future,
  // we'd like to refactor autodiff so this isn't the case. We also need an
  // optimizer for autodiff to work.
  ir.setOptimizer(ConstSGD(0.1));
  ir.updateVertices();

  // Now apply the transform.
  ir.applyTransform(Autodiff::id(), mainGraph);

  // First we check the main graph. We start by getting a test wrapper for the
  // main graph.
  IrTestWrapper tw_ir{ir};
  auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);

  // Start by checking the main graph now has a CallOp with "main_in" and
  // getGradId("main_out") inputs. This is the backward call op.
  auto tw_bwdCallOp = tw_mainGraph->ops().hasOp<CallOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().containsIds(
                   {"main_in", getGradId("main_out")})) &&
               (tw_op.outputs().hasIndex(0));
      },
      Require::MustBeTrue);

  // We will the actual CallOp* to get the called graph ID.
  auto bwdCallOp = tw_bwdCallOp->unwrap();

  // Get the #0 output of bwdCallOp (_g0).
  auto _g0 = tw_bwdCallOp->outputs().hasIndex(0)->id();

  // Find the gradsum op op that takes _g0.
  auto gradSumOp0 = tw_mainGraph->ops().hasOp<SumOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasExactIds({_g0})) &&
               (tw_op.outputs().hasIdAtIndex(0, getGradId("main_in")));
      },
      Require::MustBeTrue);

  // Now it's time to find and check subgraph bwdGraph with id "_k".
  GraphId _k       = bwdCallOp->getCalledGraph().id;
  auto tw_bwdGraph = tw_ir.hasGraph(_k, Require::MustBeTrue);

  // OK, create expected tensor IDs.
  auto &bwdGraph     = tw_bwdGraph->unwrap().get();
  auto _k_a_in       = bwdGraph.addScope("a_in");
  auto _k_a_tmp      = bwdGraph.addScope("a_tmp");
  auto _k_a_in_grad  = bwdGraph.addScope(getGradId("a_in"));
  auto _k_a_tmp_grad = bwdGraph.addScope(getGradId("a_tmp"));
  auto _k_a_out_grad = bwdGraph.addScope(getGradId("a_out"));

  // We don't actually know the values of indices _x and _y, let's determine
  // them. We know CallOps always have contiguous indices from 0 and we know
  // we have two inputs, so it's just working out which is 0 and which is 1.

  // Get the input index of "main_in" in bwdCallOp.
  auto _x =
      tw_bwdCallOp->inputs().hasId("main_in", Require::MustBeTrue)->index();
  // Get the input index of getGradId("main_out").
  auto _y = tw_bwdCallOp->inputs()
                .hasId(getGradId("main_out"), Require::MustBeTrue)
                ->index();

  // Check the subgraphs inputs/outputs.
  tw_bwdGraph->inputs().hasExactIds({_k_a_in, _k_a_out_grad});
  tw_bwdGraph->inputs().hasIdAtIndex(_x, _k_a_in);
  tw_bwdGraph->inputs().hasIdAtIndex(_y, _k_a_out_grad);
  tw_bwdGraph->outputs().hasExactIds({_k_a_in_grad});
  tw_bwdGraph->outputs().hasIdAtIndex(0, _k_a_in_grad);

  // Find the forward SimpleTestOp clone in subgraph bwdGraph that consumes
  // _k/"a_in" and produces _k/"a_tmp".
  auto tw_simpleTestOpClone = tw_bwdGraph->ops().hasOp<SimpleTestOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasIdAtIndex(0, _k_a_in)) &&
               (tw_op.outputs().hasIdAtIndex(0, _k_a_tmp));
      },
      Require::MustBeTrue);

  // Find the SimpleTestGradOp in subgraph _k that consumes _k/"a_tmp" and
  // _k/getGradId("a_out") and produces _g1.
  auto tw_simpleTestGradOp1 = tw_bwdGraph->ops().hasOp<SimpleTestGradOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasIdAtIndex(0, _k_a_out_grad)) &&
               (tw_op.inputs().hasIdAtIndex(1, _k_a_tmp)) &&
               (tw_op.outputs().hasIndex(0));
      },
      Require::MustBeTrue);

  // Figure out what _g1 is.
  auto _g1 =
      tw_simpleTestGradOp1->outputs().hasIndex(0, Require::MustBeTrue)->id();

  // Find the gradsum op in subgraph _k that consumes _g1 and produces
  // _k/getGradId("a_tmp").
  auto tw_gradSumOp1 = tw_bwdGraph->ops().hasOp<SumOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasExactIds({_g1})) &&
               (tw_op.outputs().hasIdAtIndex(0, _k_a_tmp_grad));
      },
      Require::MustBeTrue);

  // Find the SimpleTestGradOp in subgraph _k that consumes
  // _k/"a_in" and _k/getGradId("a_tmp") and produces _g2.
  auto tw_simpleTestGradOp2 = tw_bwdGraph->ops().hasOp<SimpleTestGradOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasIdAtIndex(0, _k_a_tmp_grad)) &&
               (tw_op.inputs().hasIdAtIndex(1, _k_a_in)) &&
               (tw_op.outputs().hasIndex(0));
      },
      Require::MustBeTrue);

  // Figure out what _g2 is.
  auto _g2 =
      tw_simpleTestGradOp2->outputs().hasIndex(0, Require::MustBeTrue)->id();

  // Find the gradsum op that takes _g2 and produces _k/getGradId("a_in").
  auto tw_gradSumOp2 = tw_bwdGraph->ops().hasOp<SumOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasExactIds({_g2})) &&
               (tw_op.outputs().hasIdAtIndex(0, _k_a_in_grad));
      },
      Require::MustBeTrue);
}

BOOST_AUTO_TEST_CASE(autodiff_createBwdGraph_0) {

  // Check that the autodiff prototype that operates on a single subgraph, when
  // given `gradsProvidedForTensors` as [A/a_out] returns a graph as per the
  // below:
  //                             .........[expected autodiff result]..........
  //                             : bwdGraphId: _k                            :
  //                             : expectedInputs (any order):               :
  //                             :    - _x: (A/a_in, Fwd)                    :
  //                             :    - _y: (A/a_tmp, Fwd)                   :
  //                             :    - _z: (A/a_out, FwdGrad)               :
  //                             : expectedOutputs (any order):              :
  //                             :    - _w: (A/a_in, FwdGrad)                :
  //                             :                                           :
  //    [A/a_in]                 :                      [_k/getGradId(a_in)] :
  //      |                      :                              #0 ^         :
  // .----|-[subgraph "A"]----.  : .-------------[subgraph _k]-----|-------. :
  // |    |                   |  : |                              Sum      | :
  // |    |                   |  : |                               ^       | :
  // |    |                   |  : |                               |       | :
  // |    |                   |  : |                             [_g1]     | :
  // | #0 v                   |  : |                            #0 |       | :
  // | SimpleTestOp           |  : |                    SimpleTestGradOp   | :
  // | #0 |                   |  : |                    #1 ^    #0 ^       | :
  // |    |                   |  : |                       |       |       | :
  // |    |                   |  : |     .-----------------'       |       | :
  // |  [A/a_tmp]             |  : |     |                         |       | :
  // |    |                   |  : |     |         [_k/getGradId(a_tmp)]   | :
  // |    |                   |  : |     |                         ^       | :
  // |    |                   |  : |     |                         |       | :
  // |    |                   |  : |     |                        Sum      | :
  // |    |                   |  : |     |                         ^       | :
  // |    |                   |  : |     |                         |       | :
  // |    |                   |  : |     |                       [_g0]     | :
  // | #0 v                   |  : |     |                      #0 |       | :
  // | SimpleTestOp           |  : |     |              SimpleTestGradOp   | :
  // | #0 |                   |  : |     |              #1 ^    #0 ^       | :
  // |    |                   |  : |     |                 |       |       | :
  // |    |                   |  : |     |          .------'       |       | :
  // '----|-------------------'  : '-----|----------|--------------|-------' :
  //      v                      :   #_x |      #_y |          #_z |         :
  //    [A/a_out]                :       |          |              |         :
  //                             : [_k/a_in] [_k/a_tmp] [_k/getGradId(a_out)]:
  //                             :...........................................:
  //
  // Where:
  // - _k              is an arbitrary GraphId.
  // - _w              is an arbitrary OutIndex.
  // - _x, _y, _z      are arbitrary InIndices.
  // - _g0, _g1        are arbitrary edge gradient TensorIds.

  Ir ir;

  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  int32_t tData[] = {5};

  // Create the subgraph.
  auto &mainGraph = ir.getMainGraph();
  auto &subgraphA = ir.createGraph(GraphId("A"));

  // Create subgraph A.
  auto a_in  = subgraphA.addScope("a_in");
  auto a_tmp = subgraphA.addScope("a_tmp");
  auto a_out = subgraphA.addScope("a_out");
  subgraphA.addInput(a_in, tInfo);

  // Add SimpleTestOp0.
  Op::Settings settingsSubgraphA = Op::Settings{subgraphA, "SimpleTestOp"};
  auto simpleTestOp0 = subgraphA.createOp<SimpleTestOp>(settingsSubgraphA);

  // Connect SimpleTestOp0.
  simpleTestOp0->connectInTensor(0, a_in);
  simpleTestOp0->createAndConnectOutTensor(0, a_tmp);
  simpleTestOp0->setup();

  // Add SimpleTestOp1.
  auto simpleTestOp1 = subgraphA.createOp<SimpleTestOp>(settingsSubgraphA);

  // Connect SimpleTestOp1.
  simpleTestOp1->connectInTensor(0, a_tmp);
  simpleTestOp1->createAndConnectOutTensor(0, a_out);
  simpleTestOp1->setup();

  // Mark "a_out" as a graph output.
  subgraphA.markAsOutput(a_out);

  // Now apply the createBwdGraph function.

  Autodiff autodiff;
  auto result = autodiff.createBwdGraph(std::ref(ir),
                                        GraphId("A"),
                                        Autodiff::TensorIds({a_out}),
                                        Autodiff::TensorIds({}),
                                        FwdGraphToBwdGraphInfo());

  // Now it's time to find and check subgraph bwdGraph with id "_k".
  IrTestWrapper tw_ir{ir};
  GraphId _k       = result.bwdGraphId;
  auto tw_bwdGraph = tw_ir.hasGraph(_k, Require::MustBeTrue);
  auto &bwdGraph   = tw_bwdGraph->unwrap().get();

  // Create some tensors IDs we expect.
  auto _k_a_in       = bwdGraph.addScope("a_in");
  auto _k_a_tmp      = bwdGraph.addScope("a_tmp");
  auto _k_a_out      = bwdGraph.addScope("a_out");
  auto _k_a_in_grad  = bwdGraph.addScope(getGradId("a_in"));
  auto _k_a_tmp_grad = bwdGraph.addScope(getGradId("a_tmp"));
  auto _k_a_out_grad = bwdGraph.addScope(getGradId("a_out"));

  // We don't actually know the values of indices _x, _y, _z. Let's check
  // the inputs are what we expect and determine their values.
  tw_bwdGraph->inputs().hasExactIds({_k_a_in, _k_a_tmp, _k_a_out_grad},
                                    Require::MustBeTrue);
  auto _x = tw_bwdGraph->inputs().hasId(_k_a_in, Require::MustBeTrue)->index();
  auto _y = tw_bwdGraph->inputs().hasId(_k_a_tmp, Require::MustBeTrue)->index();
  auto _z =
      tw_bwdGraph->inputs().hasId(_k_a_out_grad, Require::MustBeTrue)->index();

  // We do know there is only one output, and that output is at index 0.
  tw_bwdGraph->outputs().hasExactIds({_k_a_in_grad}, Require::MustBeTrue);

  // Check that the result.expectedInputs matches the indices we determined.
  BwdGraphInfo expectedBwdGraphInfo{
      _k,
      sortExpectedConnections({{_x, {a_in, ExpectedConnectionType::Fwd}},
                               {_y, {a_tmp, ExpectedConnectionType::Fwd}},
                               {_z, {a_out, ExpectedConnectionType::FwdGrad}}}),
      sortExpectedConnections({{0, {a_in, ExpectedConnectionType::FwdGrad}}})};

  BOOST_REQUIRE_MESSAGE(
      expectedBwdGraphInfo == result,
      logging::format("Expected {}, got {}", expectedBwdGraphInfo, result));

  // Now check the bwd graph is as expected.

  // Find the SimpleTestGradOp in subgraph _k that consumes _k/"a_tmp" and
  // _k/getGradId("a_out") and produces _g0.
  auto tw_simpleTestGradOp1 = tw_bwdGraph->ops().hasOp<SimpleTestGradOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasIdAtIndex(0, _k_a_out_grad)) &&
               (tw_op.inputs().hasIdAtIndex(1, _k_a_tmp)) &&
               (tw_op.outputs().hasIndex(0));
      },
      Require::MustBeTrue);

  // Figure out what _g0 is.
  auto _g0 =
      tw_simpleTestGradOp1->outputs().hasIndex(0, Require::MustBeTrue)->id();

  // Find the gradsum op in subgraph _k that consumes _g0 and produces
  // _k/getGradId("a_tmp").
  auto tw_gradSumOp1 = tw_bwdGraph->ops().hasOp<SumOp>(
      [&](auto &tw_op) -> bool {
        Op *op = tw_op.unwrap();
        return (tw_op.inputs().hasExactIds({_g0})) &&
               (tw_op.outputs().hasIdAtIndex(0, _k_a_tmp_grad));
      },
      Require::MustBeTrue);

  // Find the SimpleTestGradOp in subgraph _k that consumes
  // _k/"a_in" and _k/getGradId("a_tmp") and produces _g1.
  auto tw_simpleTestGradOp2 = tw_bwdGraph->ops().hasOp<SimpleTestGradOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasIdAtIndex(0, _k_a_tmp_grad)) &&
               (tw_op.inputs().hasIdAtIndex(1, _k_a_in)) &&
               (tw_op.outputs().hasIndex(0));
      },
      Require::MustBeTrue);

  // Figure out what _g1 is.
  auto _g1 =
      tw_simpleTestGradOp2->outputs().hasIndex(0, Require::MustBeTrue)->id();

  // Find the gradsum op that takes _g2 and produces _k/getGradId("a_in").
  auto tw_gradSumOp2 = tw_bwdGraph->ops().hasOp<SumOp>(
      [&](auto &tw_op) -> bool {
        return (tw_op.inputs().hasExactIds({_g1})) &&
               (tw_op.outputs().hasIdAtIndex(0, _k_a_in_grad));
      },
      Require::MustBeTrue);
}
