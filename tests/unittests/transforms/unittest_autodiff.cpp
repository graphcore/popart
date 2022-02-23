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
#include <popart/op/greater.hpp>
#include <popart/op/if.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/sum.hpp>
#include <popart/sgd.hpp>
#include <popart/tensornames.hpp>
#include <popart/util.hpp>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
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
class AdvTest1Op;
class AdvTest1GradOp;
class AdvTest2Op;
class AdvTest2GradOp;

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
 * An op with the following behaviour:
 *  - an input #0: (details unimportant)
 *  - an input #1: (details unimportant)
 *  - an output #0: (same tensor info as input #0)
 *  - calls to getGradOps() return {AdvTest1GradOp}
 *  - otherwise default popart::Op behaviour
 **/
class AdvTest1Op : public TestOp {
public:
  AdvTest1Op(const Op::Settings &settings)
      : TestOp(OperatorIdentifier("TestOps", "AdvTest1Op", 1), settings) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<AdvTest1Op>(*this);
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> grads;
    grads.emplace_back(std::make_unique<AdvTest1GradOp>(*this));
    return grads;
  }
};

/**
 * An op with the following behaviour:
 *  - an input #0: the #0 input of a AdvTest1Op
 *  - an input #1: the gradient of a AdvTest1Op's #0 output
 *  - an output #0: the gradient of a AdvTest1Op's #0 input
 *  - no output for AdvTest1Op's #1 input.
 *  - otherwise default popart::Op behaviour
 **/
class AdvTest1GradOp : public TestOp {
public:
  AdvTest1GradOp(const AdvTest1Op &op)
      : TestOp(OperatorIdentifier("TestOps", "AdvTest1GradOp", 1),
               op.Op::getSettings()) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<AdvTest1GradOp>(*this);
  }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {0, 0, GradOpInType::In}, {1, 0, GradOpInType::GradOut}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }
};

/**
 * An op with the following behaviour:
 *  - an input #0: (details unimportant)
 *  - an output #0: (same tensor info as input #0)
 *  - calls to getGradOps() return {AdvTest2GradOp}
 *  - otherwise default popart::Op behaviour
 **/
class AdvTest2Op : public TestOp {
public:
  AdvTest2Op(const Op::Settings &settings)
      : TestOp(OperatorIdentifier("TestOps", "AdvTest2Op", 1), settings) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<AdvTest2Op>(*this);
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> grads;
    grads.emplace_back(std::make_unique<AdvTest2GradOp>(*this));
    return grads;
  }
};

/**
 * An op with the following behaviour:
 *  - an input #0: the #0 input of a AdvTest2Op
 *  - an input #1: the gradient of a AdvTest2Op's #0 output
 *  - an input #2: the #0 output of a AdvTest2Op
 *  - an output #0: the gradient of a AdvTest2Op's #0 input
 *  - otherwise default popart::Op behaviour
 **/
class AdvTest2GradOp : public TestOp {
public:
  AdvTest2GradOp(const AdvTest2Op &op)
      : TestOp(OperatorIdentifier("TestOps", "AdvTest2GradOp", 1),
               op.Op::getSettings()) {}

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<AdvTest2GradOp>(*this);
  }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {0, 0, GradOpInType::In},
        {1, 0, GradOpInType::GradOut},
        {2, 0, GradOpInType::Out}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {{0, 0}};
    return outInfo;
  }
};

/**
 * Receive a mapping from indices to expected connections and output a list
 * of expected connections.
 **/
ExpectedConnections
sortExpectedConnections(const std::map<int, ExpectedConnection> &map) {

  // Check we got exactly the indices, 0, 1, 2, 3 ... map.size()-1;
  std::vector<int> expectedIndices;
  for (int i = 0; i < map.size(); ++i) {
    expectedIndices.push_back(i);
  }

  std::stringstream ss;
  ss << "Expected " << map << " to have indices " << expectedIndices;

  ExpectedConnections result;
  std::set<int> seenIndices;
  result.resize(map.size());
  for (const auto &kv : map) {
    // Check each index is in the expected range.
    BOOST_REQUIRE_MESSAGE(kv.first >= 0, ss.str());
    BOOST_REQUIRE_MESSAGE(kv.first < map.size(), ss.str());
    // Add to the result.
    result.at(kv.first) = kv.second;
    // Add to set so we can check we have each index.
    seenIndices.insert(kv.first);
  }

  // Unless we've accounted for each index this won't be true.
  BOOST_REQUIRE_MESSAGE(seenIndices.size() == map.size(), ss.str());

  return result;
}

/**
 * Add the following to an IR:
 *
 * .--[main graph]-----.
 * |                   |
 * |    ["t0"]         |
 * |      |            |
 * |   #0 v            |
 * | SimpleTestOp      |
 * |   #0 |            |
 * |      v            |
 * |    ["t1"]         |
 * |                   |
 * '-------------------'
 **/
void addTestIr1(Ir &ir) {

  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp. Connect "t0" to SimpleTestOp's input, create the "t1"
  // output and call setup().
  mainGraph.createConnectedOp<SimpleTestOp>({{0, "t0"}}, {{0, "t1"}}, settings);
}

/**
 * Add the following to an IR:
 *
 * .--[main graph]----------.
 * |                        |
 * |  ["main_in"]           |
 * |      |                 |
 * |   #0 v                 |
 * |  CallOp<A>             |
 * |   #0 |                 |
 * |      v                 |
 * |  ["main_out"]          |
 * '------------------------'
 *
 *    [A/a_in]
 *      |
 * .----|-[subgraph "A"]----.
 * | #0 v                   |
 * | SimpleTestOp           |
 * | #0 |                   |
 * |    |                   |
 * |  [A/a_tmp]             |
 * |    |                   |
 * | #0 v                   |
 * | SimpleTestOp           |
 * | #0 |                   |
 * |    |                   |
 * |    |                   |
 * '----|-------------------'
 *      v
 *    [A/a_out]
 **/
void addTestIr2(Ir &ir) {
  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  int32_t tData[] = {5};

  // Create the subgraph.
  auto &subgraphA = ir.createGraph(GraphId("A"));

  // Create subgraph A.
  auto a_in  = addScope(subgraphA, "a_in");
  auto a_tmp = addScope(subgraphA, "a_tmp");
  auto a_out = addScope(subgraphA, "a_out");
  subgraphA.addInput(a_in, tInfo);

  // Add SimpleTestOp0.
  Op::Settings settingsSubgraphA = Op::Settings{subgraphA, "SimpleTestOp"};
  subgraphA.createConnectedOp<SimpleTestOp>(
      {{0, a_in}}, {{0, a_tmp}}, settingsSubgraphA);

  // Add SimpleTestOp1.
  subgraphA.createConnectedOp<SimpleTestOp>(
      {{0, a_tmp}}, {{0, a_out}}, settingsSubgraphA);

  // Mark "a_out" as a graph output.
  subgraphA.markAsOutput(a_out);
  // Create the subgraph.
  auto &mainGraph = ir.getMainGraph();

  // Create main graph.
  mainGraph.getTensors().addVarInit(
      "main_in", tInfo, static_cast<void *>(&tData));

  mainGraph.createConnectedOp<CallOp>(
      {{0, "main_in"}},
      {{0, "main_out"}},
      Onnx::AiGraphcore::OpSet1::Call,
      std::ref(subgraphA),
      Op::Settings{mainGraph, "", mainGraph.getScope()});
}

/**
 * Add the following to an IR:
 *
 *     [A/a_in0] [A/a_in1]
 *        |        |
 * .-["A"]|--------|---------.
 * |      |        |         |
 * |      | .------'         |
 * |      | |                |
 * |      | |                |
 * |      | |                |
 * |   #0 v v #1             |
 * |   AdvTest1Op            |
 * |   #0 |                  |
 * |      |                  |
 * '------|- ----------------'
 *        |
 *        v
 *     [A/a_out0]
 **/
void addTestIr3(Ir &ir) {
  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};

  // Create the subgraph.
  auto &subgraphA = ir.createGraph(GraphId("A"));

  // Create subgraph A.
  auto a_in0  = addScope(subgraphA, "a_in0");
  auto a_in1  = addScope(subgraphA, "a_in1");
  auto a_out0 = addScope(subgraphA, "a_out0");
  subgraphA.addInput(a_in0, tInfo);
  subgraphA.addInput(a_in1, tInfo);

  // Add AdvTest1Op.
  Op::Settings settingsSubgraphA = Op::Settings{subgraphA, "AdvTest1Op"};
  subgraphA.createConnectedOp<AdvTest1Op>(
      {{0, a_in0}, {1, a_in1}}, {{0, a_out0}}, settingsSubgraphA);

  // Mark "a_out" as a graph output.
  subgraphA.markAsOutput(a_out0);
}

/**
 * Add the following to an IR:
 *
 * .--[main graph]----------.
 * |                        |
 * |  ["main_in"]           |
 * |      |                 |
 * |   #0 v                 |
 * |  CallOp<A>             |
 * |   #0 |                 |
 * |      v                 |
 * |  ["main_out"]          |
 * '------------------------'
 *
 *    [A/in]
 *      |
 * .----|-[subgraph "A"]----.
 * | #0 v                   |
 * | AdvTest2Op             |
 * | #0 |                   |
 * |    |                   |
 * |  [A/tmp]               |
 * |    |                   |
 * | #0 v                   |
 * | AdvTest2Op             |
 * | #0 |                   |
 * |    |                   |
 * |    |                   |
 * '----|-------------------'
 *      v
 *    [A/out]
 **/
void addTestIr4(Ir &ir) {
  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  int32_t tData[] = {5};

  // Create the subgraph.
  auto &subgraphA = ir.createGraph(GraphId("A"));

  // Create subgraph A.
  auto in  = addScope(subgraphA, "in");
  auto tmp = addScope(subgraphA, "tmp");
  auto out = addScope(subgraphA, "out");
  subgraphA.addInput(in, tInfo);

  // Add SimpleTestOp0.
  Op::Settings settingsSubgraphA = Op::Settings{subgraphA, "AdvTest2Op"};
  subgraphA.createConnectedOp<AdvTest2Op>(
      {{0, in}}, {{0, tmp}}, settingsSubgraphA);

  // Add SimpleTestOp1.
  subgraphA.createConnectedOp<AdvTest2Op>(
      {{0, tmp}}, {{0, out}}, settingsSubgraphA);

  // Mark "a_out" as a graph output.
  subgraphA.markAsOutput(out);
  // Create the subgraph.
  auto &mainGraph = ir.getMainGraph();

  // Create main graph.
  mainGraph.getTensors().addVarInit(
      "main_in", tInfo, static_cast<void *>(&tData));

  mainGraph.createConnectedOp<CallOp>(
      {{0, "main_in"}},
      {{0, "main_out"}},
      Onnx::AiGraphcore::OpSet1::Call,
      std::ref(subgraphA),
      Op::Settings{mainGraph, "", mainGraph.getScope()});
}

/**
 * Add the following to an IR:
 *
 * .--[main graph]----------.
 * |                        |
 * |   8       ["main_in"]  |
 * |   |           |        |
 * |   |    .------|        |
 * |   |    |      |        |
 * |#1 v #0 v      |        |
 * |  Greater      |        |
 * |     |         |        |
 * |  #0 v      #1 v        |
 * |  IfOp<Then, Else>      |
 * |   #0 |                 |
 * |      v                 |
 * |  ["main_out"]          |
 * '------------------------'
 *
 *    [Then/in]                   [Else/in]
 *      |                            |
 * .----|-["Then"]----------.   .----|-["Else"]----------.
 * | #0 v                   |   | #0 v                   |
 * | AdvTest2Op             |   | AdvTest2Op             |
 * | #0 |                   |   | #0 |                   |
 * |    |                   |   |    |                   |
 * |  [Then/tmp]            |   |  [Else/tmp]            |
 * |    |                   |   |    |                   |
 * | #0 v                   |   | #0 v                   |
 * | AdvTest2Op             |   | AdvTest2Op             |
 * | #0 |                   |   | #0 |                   |
 * |    |                   |   |    |                   |
 * |    |                   |   |    |                   |
 * '----|-------------------'   '----|-------------------'
 *      v                            v
 *    [Then/out]                   [Else/out]
 **/
void addTestIr5(Ir &ir) {
  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  int32_t tDataZero[]  = {0};
  int32_t tDataEight[] = {8};

  auto addBranch = [&](GraphId branchName) {
    // Create the subgraph.
    auto &branch = ir.createGraph(branchName);

    // Create subgraph.
    auto in  = addScope(branch, "in");
    auto tmp = addScope(branch, "tmp");
    auto out = addScope(branch, "out");
    branch.addInput(in, tInfo);

    // Add SimpleTestOp0.
    Op::Settings branchSettings = Op::Settings{branch, ""};
    branch.createConnectedOp<AdvTest2Op>({{0, in}}, {{0, tmp}}, branchSettings);

    // Add SimpleTestOp1.
    branch.createConnectedOp<AdvTest2Op>(
        {{0, tmp}}, {{0, out}}, branchSettings);

    // Mark "a_out" as a graph output.
    branch.markAsOutput(out);
  };

  addBranch(GraphId("Then"));
  addBranch(GraphId("Else"));

  // Create the subgraph.
  auto &mainGraph        = ir.getMainGraph();
  auto mainGraphSettings = Op::Settings{mainGraph, "", mainGraph.getScope()};

  // Create some tensors.
  mainGraph.getTensors().addVarInit(
      "main_in", tInfo, static_cast<void *>(&tDataZero));
  mainGraph.getTensors().addConstInit(
      "main_const", tInfo, static_cast<void *>(&tDataEight));

  mainGraph.createConnectedOp<GreaterOp>({{0, "main_in"}, {1, "main_const"}},
                                         {{0, "main_cond"}},
                                         Onnx::Operators::Greater_1,
                                         mainGraphSettings);

  mainGraph.createConnectedOp<IfOp>(
      {{0, "main_cond"}, {1, "main_in"}},
      {{0, "main_out"}},
      Onnx::Operators::If_1,
      BranchInfo{GraphId("Then"), {{1, 0}}, {{0, 0}}},
      BranchInfo{GraphId("Else"), {{1, 0}}, {{0, 0}}},
      mainGraphSettings);
}

} // namespace

BOOST_AUTO_TEST_CASE(autodiff_0) {

  // Check that when given the left hand side of the IR below (from makeTestIr1)
  // autodiff's (`bool apply(Ir&)`) adds the parts of the IR on the right:
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
  addTestIr1(ir);

  // Set "t1" as the loss.
  ir.setFinalLoss("t1");
  ir.getMainGraph().setLoss("t1");

  // NOTE: We need to do a few boilerplate things to apply autodiff. In future,
  // we'd like to refactor autodiff so this isn't the case. We also need an
  // optimizer for autodiff to work.
  ir.setOptimizer(ConstSGD(0.1));
  ir.updateVertices();

  // Now apply the transform.
  ir.applyTransform(Autodiff::id(), ir.getMainGraph());

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

  // Check that autodiff (`bool apply(Ir&)`) applied to the IR generated by
  // `testIr2` adds parts of the IR marked on the right-hand side below:
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
  addTestIr2(ir);

  // Set "t1" as the loss.
  ir.setFinalLoss("main_out");
  ir.getMainGraph().setLoss("main_out");

  // NOTE: We need to do a few boilerplate things to apply autodiff. In future,
  // we'd like to refactor autodiff so this isn't the case. We also need an
  // optimizer for autodiff to work.
  ir.setOptimizer(ConstSGD(0.1));
  ir.updateVertices();

  // Now apply the transform.
  ir.applyTransform(Autodiff::id(), ir.getMainGraph());

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
  auto _k_a_in       = addScope(bwdGraph, "a_in");
  auto _k_a_tmp      = addScope(bwdGraph, "a_tmp");
  auto _k_a_in_grad  = addScope(bwdGraph, getGradId("a_in"));
  auto _k_a_tmp_grad = addScope(bwdGraph, getGradId("a_tmp"));
  auto _k_a_out_grad = addScope(bwdGraph, getGradId("a_out"));

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

  // Check that the autodiff prototype that operates on subgraph "A" of testIr2
  // `gradsProvidedForTensors` as [A/a_out] returns a graph as per the
  // right-hand side below:
  //
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
  addTestIr2(ir);

  // Get some tensor ids.
  auto &subgraphA = ir.getGraph(GraphId("A"));
  auto a_in       = addScope(subgraphA, "a_in");
  auto a_tmp      = addScope(subgraphA, "a_tmp");
  auto a_out      = addScope(subgraphA, "a_out");

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
  auto _k_a_in       = addScope(bwdGraph, "a_in");
  auto _k_a_tmp      = addScope(bwdGraph, "a_tmp");
  auto _k_a_out      = addScope(bwdGraph, "a_out");
  auto _k_a_in_grad  = addScope(bwdGraph, getGradId("a_in"));
  auto _k_a_tmp_grad = addScope(bwdGraph, getGradId("a_tmp"));
  auto _k_a_out_grad = addScope(bwdGraph, getGradId("a_out"));

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
        tw_op.unwrap();
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

BOOST_AUTO_TEST_CASE(autodiff_createBwdGraph_1) {

  // Test that, with subgraph A in the left-hand side of the IR below (from
  // addTestIr3):
  //
  // - If a gradient is provided for A/a_out0 and a gradient is required for
  //   for both A/a_in0 and A/a_in1 then an error is raised (because
  //   AdvTest1GradOp can't produce a gradient for that input).
  // - If no gradient is provided and a gradient is required for A/a_in0 then
  //   an error is raised (because AdvTest1GradOp needs the a_out0 gradient
  //   input to produce the gradient for a_in0).
  // - If a gradient is provided for A/a_out0 and a gradient is required for
  //   A/a_in0 then all is well and a gradient for A/a_in0 is available.
  //
  //                             ....[generated by autodiff on success]........
  //                             :                                            :
  //     [A/a_in0] [A/a_in1]     :  [_k/getGradId(a_in0)]                     :
  //        |        |           :        ^                                   :
  // .-["A"]|--------|---------. : .-[_k]-|---------------------------------. :
  // |      |        |         | : |      |                                 | :
  // |      | .------'         | : |      |       .. No output for a_in1's  | :
  // |      | |                | : |      |      :   gradient here, so if   | :
  // |      | |                | : |      |      :   the user requires this | :
  // |      | |                | : |      |      :   it's an error.         | :
  // |   #0 v v #1             | : |   #0 |      :                          | :
  // |   AdvTest1Op            | : |   AdvTest1GradOp                       | :
  // |   #0 |                  | : |   #0 ^   #1 ^                          | :
  // |      |                  | : |      |      |.. If this gradient is not| :
  // |      |                  | : |  .---'      |   provided then the grad | :
  // |      |                  | : |  |          |   for a_in0 can't be     | :
  // |      |                  | : |  |          |   created.               | :
  // '------|- ----------------' : '--|----------|--------------------------' :
  //        |                    :    |          |                            :
  //        v                    :    |          |                            :
  //     [A/a_out0]              :  [_k/a_in0] [_k/getGradId(a_out0)]         :
  //                             :                                            :
  //                             :............................................:

  Ir ir;
  addTestIr3(ir);

  // Get tensorIds.
  auto &subgraphA = ir.getGraph(GraphId("A"));
  auto a_in0      = addScope(subgraphA, "a_in0");
  auto a_in1      = addScope(subgraphA, "a_in1");
  auto a_out0     = addScope(subgraphA, "a_out0");

  auto test = [&](Autodiff::TensorIds provided, Autodiff::TensorIds required) {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Testing provided=" << provided << " required=" << required
              << std::endl;

    // Now call `createBwdGraph`.
    Autodiff autodiff;
    auto result = autodiff.createBwdGraph(std::ref(ir),
                                          GraphId("A"),
                                          provided,
                                          required,
                                          FwdGraphToBwdGraphInfo());

    std::cout << "Got bwdGraphId=" << result.bwdGraphId << std::endl;

    ir.logIr();
  };

  // If a gradient is provided for A/a_out0 and a gradient is required for
  // for both A/a_in0 and A/a_in1 then an error is raised (because
  // AdvTest1GradOp can't produce a gradient for that input).
  BOOST_REQUIRE_THROW(test({a_out0}, {a_in0, a_in1}), std::runtime_error);

  // If no gradient is provided and a gradient is required for A/a_in0 then
  // an error is raised (because AdvTest1GradOp needs the a_out0 gradient
  // input to produce the gradient for a_in0).
  BOOST_REQUIRE_THROW(test({}, {a_in0}), std::runtime_error);

  // If a gradient is provided for A/a_out0 and a gradient is required for
  // A/a_in0 then all is well and a gradient for A/a_in0 is available.
  BOOST_REQUIRE_NO_THROW(test({a_out0}, {a_in0}));
}

BOOST_AUTO_TEST_CASE(autodiff_stitch_0) {

  // Test that, with the following IRs (obtained via addTestIr4 for
  // CallOpTestType and addTestIr5 for IfOpTestType and createBwdGraph) and
  // using specific stitch parameters, we get the expected outputs.
  //
  // .--[main graph]----------.
  // |                        |    .--- CallOp for CallOpTestType
  // |     ...                |    |    IfOp   for IfOpTestType
  // |      |                 |    |
  // |      v                 |    |
  // |  Op<_j, ...>  *--------(----'
  // |   #0 |                 |
  // |      v                 |
  // |  ["main_out"]          |
  // '------------------------'
  //
  //      .---------------------------- _j="A"    for CallOpTestType
  //      |                             _j="Then" for IfOpTestType
  //      |
  //      *
  //
  //    [_j/in]                                         [_k/getGradId(in)]
  //      |                                                     #0 ^
  // .----|-[subgraph _j]-----.    .-------------[subgraph _k]-----|-------.
  // |    |                   |    |                              Sum      |
  // |    |                   |    |                               ^       |
  // |    |                   |    |                               |       |
  // |    |                   |    |                             [_g1]     |
  // | #0 v                   |    |                            #0 |       |
  // | AdvTest2Op             |    |                      AdvTest2GradOp   |
  // | #0 |                   |    |                    #1 ^  #2 ^ ^ #0    |
  // |    |                   |    |                       |     | |       |
  // |    |                   |    |     .-----------------'     | |       |
  // |  [A/tmp]               |    |     |        .--------------' |       |
  // |    |                   |    |     |        |                |       |
  // |    |                   |    |     |        |  [_k/getGradId(a_tmp)] |
  // |    |                   |    |     |        |                ^       |
  // |    |                   |    |     |        |                |       |
  // |    |                   |    |     |        |               Sum      |
  // |    |                   |    |     |        |                ^       |
  // |    |                   |    |     |        |                |       |
  // |    |                   |    |     |        |              [_g0]     |
  // | #0 v                   |    |     |        |             #0 |       |
  // | AdvTest2Op             |    |     |        |       AdvTest2GradOp   |
  // | #0 |                   |    |     |        |    #1 ^  #2 ^  ^ #0    |
  // |    |                   |    |     |        |       |     |  |       |
  // |    |                   |    |     |        |-------'     |  '-----. |
  // |    |                   |    |     |        |             |        | |
  // '----|-------------------'    '-----|--------|-------------|--------|-'
  //      v                          #_w |    #_x |         #_y |    #_z |
  //    [_j/out]                         |        |             |        |
  //                               [_k/a_in]   [_k/tmp]      [_k/out]    |
  //                                                     [_k/getGradId(out)]
  //
  // Now, if stitch is called on _k we expect the following result IRs:
  //
  // .--[main graph]----------------.
  // |                              |  C1: Extra output at call site and extra
  // |      ...                     |      subgraph output. Only applies for
  // |       |                      |      CallOpTestType, stitch strategy
  // | ......(..................... |      AddFwdOutputs and when _x is in the
  // | :  #0 v                 C1 : |      stitch index list. Using defaults,
  // | : Op<_j, ...> -------.     : |      this stitch index should be included.
  // | :  #0 |              |     : |  C2: _k/tmp input replaced by
  // | :.....(.........     |     : |      recomputation. Only with recompute
  // |       |        :     |     : |      stitch strategies and when _x is in
  // |       v        :     v     : |      the stitch index list. Using
  // |   ["main_out"] :  ["tmp"]  : |      defaults, this index is included for
  // |                :...........: |      both recomputation strategies.
  // '------------------------------'  C3: _k/out input replaced by
  //                                       recomputation. Only with recompute
  //                                       stitch strategies and when _y is in
  //                                       the stitch index list. Using
  //                                       defaults, this index is included
  //                                       with RecomputeAllNonInputs but not
  //                                       with RecomputeMinimal, as it is
  //                                       an output of _j, so stitching is not
  //                                       required.
  //
  //                                       NOTE: SafeAddFwdOutputs is a hybrid
  //                                       of RecomputeMinmal and AddFwdOutputs.
  //
  //
  //
  //    [_j/in]                                           [_k/getGradId(in)]
  //      |                                                      #0 ^
  // .----|-[subgraph _j]------.    .-------------[subgraph _k]-----|-------.
  // |    |                    |    |                              Sum      |
  // |    |                    |    |                               ^       |
  // |    |                    |    |                               |       |
  // |    |                    |    |                             [_g1]     |
  // | #0 v                    |    |                            #0 |       |
  // | AdvTest2Op              |    |                      AdvTest2GradOp   |
  // | #0 |                    |    |                    #1 ^  #2 ^ ^ #0    |
  // |    |                    |    |                       |     | |       |
  // |  [_j/tmp]               |    |     .-----------------'     | |       |
  // |  ..(...............     |    |     |        .--------------' |       |
  // |  : |           C1 :     |    |     |        |                |       |
  // |  : |              :     |    |     |        |  [_k/getGradId(a_tmp)] |
  // |  : |------------. :     |    |     |        |                ^       |
  // |  : |            | :     |    |     |        |                |       |
  // |  :.(..........  | :     |    |     |        |               Sum      |
  // |    |          : | :     |    |     |        |                ^       |
  // | #0 v          : | :     |    |     |        |                |       |
  // | AdvTest2Op    : | :     |    |     |        |              [_g0]     |
  // | #0 |          : | :     |    |     |        |             #0 |       |
  // |    |      ....: | :.... |    |     |        |       AdvTest2GradOp   |
  // |    |      :     |     : |    |     |        |    #1 ^  #2 ^  ^ #0    |
  // |    |      :     |     : |    |     |        |       |     |  |       |
  // |    |      :     |     : |    |     |        |-------'     |  '-----. |
  // |    |      :     |     : |    |     |        |          [k_out]     | |
  // |    |      :     |     : |    |     |        |             |        | |
  // |    |      :     |     : |    |     |      ..(.............(...     | |
  // |    |      :     |     : |    |     |      : |          #0 |  :     | |
  // |    |      :     |     : |    |     |      : |     AdvTest2Op :     | |
  // |    |      :     |     : |    |     |      : |          #0 ^  :     | |
  // |    |      :     |     : |    |     |      : |-------------'  :     | |
  // |    |      :     |     : |    |     |      :.(........        :     | |
  // |    |      :     |     : |    |     |        |       :        :     | |
  // |    |      :     |     : |    |     |     [_k/tmp]   :        :     | |
  // |    |      :     |     : |    |     |        |       :        :     | |
  // |    |      :     |     : |    |   ..(........(.....  :        :     | |
  // |    |      :     |     : |    |   : | #0     |    :  :        :     | |
  // |    |      :     |     : |    |   : | AdvTest2Op  :  :        :     | |
  // |    |      :     |     : |    |   : |     #0 ^    :  :        :     | |
  // |    |      :     |     : |    |   : |--------'    :  :        :     | |
  // |    |      :     |     : |    |   :.)....         :  :        :     | |
  // '----|------:-----|-----(-'    '-----|---(---------(--(--------(-----|-'
  //      v      :     v     :        #_a |   : (#_b)   :  : (#_c)  : #_d |
  //   [_j/out]  : [_j/tmp]  :        [_k/in] :      C2 :  :     C3 :     |
  //             :...........:                :.........:  :........:     |
  //                                                      [_k/getGradId(out)

  enum class TestType { CallOpTestType = 0, IfOpTestType = 1 };

  enum class ExpectC1 { Yes = 0, No = 1 };

  enum class ExpectC2 { Yes = 0, No = 1 };

  enum class ExpectC3 { Yes = 0, No = 1 };

  struct IrInfo {
    // GraphId values.
    GraphId _j{""};
    GraphId _k{""};

    // Main graph tensor names.
    TensorId in;
    TensorId tmp;
    TensorId out;

    // Backward graph tensor names.
    TensorId _k_in;
    TensorId _k_tmp;
    TensorId _k_out;
    TensorId _k_in_grad;
    TensorId _k_tmp_grad;
    TensorId _k_out_grad;

    // Backward graph input indices (before stitching).
    InIndex _w;
    InIndex _x;
    InIndex _y;
    InIndex _z;

    // Backward graph input indices (after stitching, -1 if not available).
    InIndex _a;
    InIndex _b;
    InIndex _c;
    InIndex _d;
  };

  using OptStitchIndices = nonstd::optional<std::vector<InIndex>>;

  auto test = [&](TestType testType,
                  AutodiffStitchStrategy strategy,
                  // Generate stitchIndices.
                  std::function<OptStitchIndices(IrInfo)> stitchIndicesFun,
                  // Generate expected BwdGraphInfo object.
                  std::function<BwdGraphInfo(IrInfo)> expectedBwdGraphInfoFun,
                  ExpectC1 c1,
                  ExpectC2 c2,
                  ExpectC3 c3) {
    Ir ir;
    IrInfo irInfo;
    if (testType == TestType::CallOpTestType) {
      addTestIr4(ir);
      irInfo._j = GraphId("A");
    } else if (testType == TestType::IfOpTestType) {
      addTestIr5(ir);
      irInfo._j = GraphId("Then");
    }

    // Get some info about IR.
    auto &fwdGraph = ir.getGraph(irInfo._j);
    irInfo.in      = addScope(fwdGraph, "in");
    irInfo.tmp     = addScope(fwdGraph, "tmp");
    irInfo.out     = addScope(fwdGraph, "out");

    // Now apply the createBwdGraph function on _j to generate _k.
    Autodiff autodiff;
    auto createBwdGraphResult =
        autodiff.createBwdGraph(std::ref(ir),
                                irInfo._j,
                                Autodiff::TensorIds({irInfo.out}),
                                Autodiff::TensorIds({}),
                                FwdGraphToBwdGraphInfo());

    // Do some introspection to get _y.
    IrTestWrapper tw_ir{ir};
    irInfo._k = createBwdGraphResult.bwdGraphId;
    auto tw_mainGraph =
        tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);
    auto tw_fwdGraph = tw_ir.hasGraph(irInfo._j, Require::MustBeTrue);
    auto tw_bwdGraph = tw_ir.hasGraph(irInfo._k, Require::MustBeTrue);
    auto &bwdGraph   = tw_bwdGraph->unwrap().get();

    // Create some tensors IDs we need.
    irInfo._k_in       = addScope(bwdGraph, "in");
    irInfo._k_tmp      = addScope(bwdGraph, "tmp");
    irInfo._k_out      = addScope(bwdGraph, "out");
    irInfo._k_in_grad  = addScope(bwdGraph, getGradId("in"));
    irInfo._k_tmp_grad = addScope(bwdGraph, getGradId("tmp"));
    irInfo._k_out_grad = addScope(bwdGraph, getGradId("out"));

    // Get _w, _x_, _y and _z.
    irInfo._w =
        tw_bwdGraph->inputs().hasId(irInfo._k_in, Require::MustBeTrue)->index();
    irInfo._x = tw_bwdGraph->inputs()
                    .hasId(irInfo._k_tmp, Require::MustBeTrue)
                    ->index();
    irInfo._y = tw_bwdGraph->inputs()
                    .hasId(irInfo._k_out, Require::MustBeTrue)
                    ->index();
    irInfo._z = tw_bwdGraph->inputs()
                    .hasId(irInfo._k_out_grad, Require::MustBeTrue)
                    ->index();

    // Now stitch it (the bit we're really testing).
    auto stitchResult = autodiff.stitch(ir,
                                        irInfo._j,
                                        createBwdGraphResult,
                                        strategy,
                                        stitchIndicesFun(irInfo));

    // Get _a, _b, _c and _d if available.
    auto getOptInIndex = [&](auto opt) -> InIndex {
      if (opt) {
        return opt->index();
      } else {
        return -1;
      }
    };

    irInfo._a = getOptInIndex(tw_bwdGraph->inputs().hasId(irInfo._k_in));
    irInfo._b = getOptInIndex(tw_bwdGraph->inputs().hasId(irInfo._k_tmp));
    irInfo._c = getOptInIndex(tw_bwdGraph->inputs().hasId(irInfo._k_out));
    irInfo._d = getOptInIndex(tw_bwdGraph->inputs().hasId(irInfo._k_out_grad));

    // Now we do some selective checks to test that the changes are what we
    // anticipate. For the sake of brevity we only for the presence or
    // absence of C1, C2 and C3 and the returned BwdGraphInfo.

    auto getRequire = [](bool require) {
      if (require) {
        // Failure unless it's true.
        return Require::MustBeTrue;
      } else {
        // Failure if it's true.
        return Require::MustBeFalse;
      }
    };

    // C1: Test for additional output in fwdGraph iff we expect C1.
    tw_fwdGraph->outputs().hasId(irInfo.tmp, getRequire(c1 == ExpectC1::Yes));
    // C1: Test the callop in the main graph has an additional output iff we
    // expect C1.
    tw_mainGraph->ops().hasOp<CallOp>(
        [&](auto &op) -> bool {
          return op.outputs().hasIndex(1).operator bool();
        },
        getRequire(c1 == ExpectC1::Yes));

    // C2: Test the bwd graph no longer has the input iff we expect C2.
    tw_bwdGraph->inputs().hasId(irInfo._k_tmp, getRequire(c2 == ExpectC2::No));
    // C2: Test that there is a `AdvTest2Op` in the bwdGraph now that
    // reproduces tmp from in iff we expect C2.
    tw_bwdGraph->ops().hasOp<AdvTest2Op>(
        [&](auto &op) -> bool {
          return (op.inputs().hasId(irInfo._k_in)) &&
                 (op.outputs().hasId(irInfo._k_tmp));
        },
        getRequire(c2 == ExpectC2::Yes));

    // C3: Test the bwd graph no longer has the input iff we expect C3.
    tw_bwdGraph->inputs().hasId(irInfo._k_out, getRequire(c3 == ExpectC3::No));
    // C3: Test that there is a `AdvTest2Op` in the bwdGraph now that
    // reproduces out from tmp iff we expect C3.
    tw_bwdGraph->ops().hasOp<AdvTest2Op>(
        [&](auto &op) -> bool {
          return (op.inputs().hasId(irInfo._k_tmp)) &&
                 (op.outputs().hasId(irInfo._k_out));
        },
        getRequire(c3 == ExpectC3::Yes));

    // Check stitchResult is the expected output.
    auto expectedBwdGraphInfo = expectedBwdGraphInfoFun(irInfo);
    BOOST_REQUIRE_MESSAGE(expectedBwdGraphInfo == stitchResult,
                          logging::format("Expected {}, got {}",
                                          expectedBwdGraphInfo,
                                          stitchResult));
  };

  for (auto testType : {TestType::CallOpTestType, TestType::IfOpTestType}) {

    // Test RecomputeMinimal + default stitch indices. Expect C2 only.
    test(
        testType,
        AutodiffStitchStrategy::RecomputeMinimal,
        [](IrInfo) { return nonstd::nullopt; },
        [](IrInfo irInfo) {
          return BwdGraphInfo{
              irInfo._k,
              sortExpectedConnections(
                  {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                   {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                   {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
              sortExpectedConnections(
                  {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
        },
        ExpectC1::No,
        ExpectC2::Yes,
        ExpectC3::No);

    // Test RecomputeMinimal + no stitch indices. Expect nothing.
    test(
        testType,
        AutodiffStitchStrategy::RecomputeMinimal,
        [](IrInfo) { return OptStitchIndices{{}}; },
        [](IrInfo irInfo) {
          return BwdGraphInfo{
              irInfo._k,
              sortExpectedConnections(
                  {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                   {irInfo._b, {irInfo.tmp, ExpectedConnectionType::Fwd}},
                   {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                   {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
              sortExpectedConnections(
                  {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
        },
        ExpectC1::No,
        ExpectC2::No,
        ExpectC3::No);

    // Test RecomputeMinimal + unstichable index. Expect exception.
    BOOST_REQUIRE_THROW(
        test(
            testType,
            AutodiffStitchStrategy::RecomputeMinimal,
            [](IrInfo irInfo) { return OptStitchIndices{{irInfo._d}}; },
            [](IrInfo irInfo) {
              return BwdGraphInfo{irInfo._k, {}, {}};
            },
            ExpectC1::No,
            ExpectC2::No,
            ExpectC3::No),
        std::runtime_error);

    // Test RecomputeAllNonInputs + default stitch indices. Expect C2 & C3.
    test(
        testType,
        AutodiffStitchStrategy::RecomputeAllNonInputs,
        [](IrInfo) { return nonstd::nullopt; },
        [](IrInfo irInfo) {
          return BwdGraphInfo{
              irInfo._k,
              sortExpectedConnections(
                  {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                   {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
              sortExpectedConnections(
                  {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
        },
        ExpectC1::No,
        ExpectC2::Yes,
        ExpectC3::Yes);

    // Test RecomputeAllNonInputs + _b. Expect C2 only.
    test(
        testType,
        AutodiffStitchStrategy::RecomputeAllNonInputs,
        [](IrInfo irInfo) { return OptStitchIndices{{irInfo._x}}; },
        [](IrInfo irInfo) {
          return BwdGraphInfo{
              irInfo._k,
              sortExpectedConnections(
                  {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                   {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                   {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
              sortExpectedConnections(
                  {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
        },
        ExpectC1::No,
        ExpectC2::Yes,
        ExpectC3::No);

    // Test RecomputeAllNonInputs + unstichable index. Expect exception.
    BOOST_REQUIRE_THROW(
        test(
            testType,
            AutodiffStitchStrategy::RecomputeAllNonInputs,
            [](IrInfo irInfo) { return OptStitchIndices{{irInfo._z}}; },
            [](IrInfo irInfo) {
              return BwdGraphInfo{irInfo._k, {}, {}};
            },
            ExpectC1::No,
            ExpectC2::No,
            ExpectC3::No),
        std::runtime_error);
  }

  // Test CallOp + AddFwdOutputs + default stitch indices. Expect C1.
  test(
      TestType::CallOpTestType,
      AutodiffStitchStrategy::AddFwdOutputs,
      [](IrInfo) { return nonstd::nullopt; },
      [](IrInfo irInfo) {
        return BwdGraphInfo{
            irInfo._k,
            sortExpectedConnections(
                {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                 {irInfo._b, {irInfo.tmp, ExpectedConnectionType::Fwd}},
                 {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                 {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
            sortExpectedConnections(
                {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
      },
      ExpectC1::Yes,
      ExpectC2::No,
      ExpectC3::No);

  // Test CallOp + AddFwdOutputs + unstitchable index. Expect exception.
  BOOST_REQUIRE_THROW(
      test(
          TestType::CallOpTestType,
          AutodiffStitchStrategy::AddFwdOutputs,
          [](IrInfo irInfo) { return OptStitchIndices{{irInfo._y}}; },
          [](IrInfo irInfo) {
            return BwdGraphInfo{irInfo._k, {}, {}};
          },
          ExpectC1::No,
          ExpectC2::No,
          ExpectC3::No),
      std::runtime_error);

  // Test with CallOp + AddFwdOutputs + that indices not in the stitch index
  // list don't get stitched.
  test(
      TestType::CallOpTestType,
      AutodiffStitchStrategy::AddFwdOutputs,
      [](IrInfo irInfo) { return OptStitchIndices{{}}; },
      [](IrInfo irInfo) {
        return BwdGraphInfo{
            irInfo._k,
            sortExpectedConnections(
                {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                 {irInfo._b, {irInfo.tmp, ExpectedConnectionType::Fwd}},
                 {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                 {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
            sortExpectedConnections(
                {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
      },
      ExpectC1::No,
      ExpectC2::No,
      ExpectC3::No);

  // Test IfOp + AddFwdOutputs + default stitch indices.
  // Does nothing, because this stitch method can't deal with IfOps.
  test(
      TestType::IfOpTestType,
      AutodiffStitchStrategy::AddFwdOutputs,
      [](IrInfo) { return nonstd::nullopt; },
      [](IrInfo irInfo) {
        return BwdGraphInfo{
            irInfo._k,
            sortExpectedConnections(
                {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                 {irInfo._b, {irInfo.tmp, ExpectedConnectionType::Fwd}},
                 {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                 {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
            sortExpectedConnections(
                {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
      },
      ExpectC1::No,
      ExpectC2::No,
      ExpectC3::No);

  // -

  // Test CallOp + SafeAddFwdOutputs + default stitch indices. Expect C1 only.
  test(
      TestType::CallOpTestType,
      AutodiffStitchStrategy::SafeAddFwdOutputs,
      [](IrInfo) { return nonstd::nullopt; },
      [](IrInfo irInfo) {
        return BwdGraphInfo{
            irInfo._k,
            sortExpectedConnections(
                {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                 {irInfo._b, {irInfo.tmp, ExpectedConnectionType::Fwd}},
                 {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                 {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
            sortExpectedConnections(
                {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
      },
      ExpectC1::Yes,
      ExpectC2::No,
      ExpectC3::No);

  // Test IfOp + SafeAddFwdOutputs + default stitch indices. Expect C2 because
  // we can't do C1 for IfOps.
  test(
      TestType::IfOpTestType,
      AutodiffStitchStrategy::SafeAddFwdOutputs,
      [](IrInfo) { return nonstd::nullopt; },
      [](IrInfo irInfo) {
        return BwdGraphInfo{
            irInfo._k,
            sortExpectedConnections(
                {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                 {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                 {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
            sortExpectedConnections(
                {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
      },
      ExpectC1::No,
      ExpectC2::Yes,
      ExpectC3::No);

  // Test CallOp + SafeAddFwdOutputs + unstitchable index. Expect exception.
  BOOST_REQUIRE_THROW(
      test(
          TestType::CallOpTestType,
          AutodiffStitchStrategy::SafeAddFwdOutputs,
          [](IrInfo irInfo) { return OptStitchIndices{{irInfo._z}}; },
          [](IrInfo irInfo) {
            return BwdGraphInfo{irInfo._k, {}, {}};
          },
          ExpectC1::No,
          ExpectC2::No,
          ExpectC3::No),
      std::runtime_error);

  // Test with CallOp + SafeAddFwdOutputs + that indices not in the stitch index
  // list don't get stitched.
  test(
      TestType::CallOpTestType,
      AutodiffStitchStrategy::SafeAddFwdOutputs,
      [](IrInfo irInfo) { return OptStitchIndices{{}}; },
      [](IrInfo irInfo) {
        return BwdGraphInfo{
            irInfo._k,
            sortExpectedConnections(
                {{irInfo._a, {irInfo.in, ExpectedConnectionType::Fwd}},
                 {irInfo._b, {irInfo.tmp, ExpectedConnectionType::Fwd}},
                 {irInfo._c, {irInfo.out, ExpectedConnectionType::Fwd}},
                 {irInfo._d, {irInfo.out, ExpectedConnectionType::FwdGrad}}}),
            sortExpectedConnections(
                {{0, {irInfo.in, ExpectedConnectionType::FwdGrad}}})};
      },
      ExpectC1::No,
      ExpectC2::No,
      ExpectC3::No);
}
