// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE autodiff_unittest

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/sum.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensornames.hpp>
#include <popart/transforms/autodiff.hpp>

#include <testutil/irquery/irquerier.hpp>

using namespace popart;
using namespace popart::testing;

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

} // namespace

BOOST_AUTO_TEST_CASE(autodiff_0) {

  // Test autodiff adds the following to the IR:
  //
  // .----------------[main graph]-------------------.
  // |                                               |
  // |                ....[added by autodiff]....... |
  // |                :                            : |
  // |    ["t0"]      :        [getGradId("t0")]   : |
  // |      |         :                ^           : |
  // |      |         :                |           : |
  // |      |         :               Sum          : |
  // |      |         :                ^           : |
  // |      |---------(-----.          |           : |
  // |      |         :     |         [??]         : |
  // |      |         :     |          |           : |
  // |   #0 v         :     |          | #0        : |
  // | SimpleTestOp   :     | SimpleTestGradOp     : |
  // |   #0 |         :     | #1 ^    #0 ^         : |
  // |      |         :     '----'       |         : |
  // |      v         :                  |         : |
  // |    ["t1"]      :      [getGradId("t1")]     : |
  // |                :                            : |
  // |                :                            : |
  // |                :............................: |
  // '-----------------------------------------------'
  //
  // Where t1 is marked as the loss and t0 is a variable tensor.

  Ir ir;
  IrQuerier q;

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

  // NOTE: We need to do a few boilerplate things to apply autodiff. For
  // example, we need to put the IR in a state where nEdgesToLoss is populated
  // correctly. In future, we'd like to refactor autodiff so this isn't the
  // case. We also need an optimizer for autodiff to work.
  ir.setOptimizer(ConstSGD(0.1));
  ir.updateVertices();
  ir.setNEdgesToLoss();

  // Now apply the transform.
  ir.applyTransform(Autodiff::id(), ir.getMainGraph());

  auto simpleTestGradOp = q.graphHasOp(
      ir.getMainGraph(),
      [&](Op *op) {
        return (dynamic_cast<SimpleTestGradOp *>(op) != nullptr) &&
               (q.opHasInputIdAt(op, 0, getGradId("t1")) &&
                (q.opHasInputIdAt(op, 1, "t0")) && (q.opHasOutputAt(op, 0)));
      },
      Require::MustBeTrue);

  // NOTE: The next three tests are redundant as they should already hold by
  // construction of the predicate above. We test them anyway to illustrate how
  // you could use them.

  // Check the TestGradOp has #0 input for getGradId("t1").
  q.opHasInputIdAt(simpleTestGradOp, 0, getGradId("t1"), Require::MustBeTrue);
  // Now, check the TestGradOp we found has a #1 input for t0.
  q.opHasInputIdAt(simpleTestGradOp, 1, "t0", Require::MustBeTrue);
  // Now, check the TestGradOp has a #0 output to some tensor.
  q.opHasOutputAt(simpleTestGradOp, 0, Require::MustBeTrue);

  // Get the #0 output of simpleTestGradOp.
  auto simpleTestGradOpOut0 = simpleTestGradOp->outId(0);

  // Find the gradsum op op that takes tmpId.
  auto gradSumOp = q.graphHasOp(
      ir.getMainGraph(),
      [&](Op *op) {
        return (dynamic_cast<SumOp *>(op) != nullptr);
        (q.opHasInputIds(op, {simpleTestGradOpOut0}));
      },
      Require::MustBeTrue);

  // Check the gradSumOp has *only* has simpleTestGradOpOut0 as input.
  q.opHasExactInputIds(gradSumOp, {simpleTestGradOpOut0}, Require::MustBeTrue);
  // Check the gradSumOp has #0 output is getGradId("t0").
  q.opHasOutputIdAt(gradSumOp, 0, getGradId("t0"), Require::MustBeTrue);
}
