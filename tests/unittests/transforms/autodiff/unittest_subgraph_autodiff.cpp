// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_transform_subgraph_autodiff
#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/graphid.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/init.hpp>
#include <popart/op/sum.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensor.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

#include <popart/transforms/autodiff.hpp>

#include <testutil/irquery/irquery.hpp>

#include <boost/hof/unpack.hpp>

#include <utility>

using namespace popart;

namespace {

class BaseTestOp;
class UnaryOp;
class UnaryGradOp;
class BinaryOp;
class BinaryLhsGradOp;
class BinaryRhsGradOp;
class BinaryOpDifferentiableOnLhsOnly;
class TertiaryOp;
class TertiaryGradOp;

/**
 * Base class for stub ops in this file.
 */
class BaseTestOp : public Op {
public:
  BaseTestOp(const OperatorIdentifier &opid, const Op::Settings &settings)
      : Op(opid, settings) {}

  virtual float getSubgraphValue() const override {
    return getLowSubgraphValue();
  }
};

/**
 *  - One input, one output.
 *  - Has one grad op, UnaryGradOp.
 *  - GradOp requires output and gradient of output.
 */
class UnaryOp : public BaseTestOp {
public:
  UnaryOp(const Op::Settings &settings)
      : BaseTestOp(OperatorIdentifier("TestOps", "UnaryOp", 1), settings) {}

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  void setup() override { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<UnaryOp>(*this);
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> grads;
    grads.emplace_back(std::make_unique<UnaryGradOp>(*this));
    return grads;
  }
};

class UnaryGradOp : public BaseTestOp {
public:
  UnaryGradOp(const UnaryOp &op)
      : BaseTestOp(OperatorIdentifier("TestOps", "UnaryGradOp", 1),
                   op.Op::getSettings()) {}

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArgInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void setup() override { outInfo(getOutIndex()) = inInfo(getGradInIndex()); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<UnaryGradOp>(*this);
  }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {getGradInIndex(), UnaryOp::getOutIndex(), GradOpInType::GradOut},
        {getFwdArgInIndex(), UnaryOp::getOutIndex(), GradOpInType::Out}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {
        {getOutIndex(), UnaryOp::getInIndex()}};
    return outInfo;
  }
};

namespace TestOps {
OperatorIdentifier BinaryOp("TestOps", "BinaryOp", 1);
}

/**
 *  - Two inputs, one output.
 *  - Has separate grad ops for the lhs and rhs inputs.
 *     - BinaryLhsGradOp
 *     - BinaryRhsGradOp
 *  - Both grad ops require the other side input, and the gradient of the
 *    output.
 */
class BinaryOp : public BaseTestOp {
public:
  BinaryOp(const Op::Settings &settings,
           OperatorIdentifier opid = TestOps::BinaryOp)
      : BaseTestOp(opid, settings) {}

  static InIndex getLhsInIndex() { return 0; }
  static InIndex getRhsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void setup() override { outInfo(getOutIndex()) = inInfo(getLhsInIndex()); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<BinaryOp>(*this);
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> grads;
    grads.emplace_back(std::make_unique<BinaryLhsGradOp>(*this));
    grads.emplace_back(std::make_unique<BinaryRhsGradOp>(*this));
    return grads;
  }
};

class BinaryLhsGradOp : public BaseTestOp {
public:
  BinaryLhsGradOp(const BinaryOp &op)
      : BaseTestOp(OperatorIdentifier("TestOps", "BinaryLhsGradOp", 1),
                   op.Op::getSettings()) {}

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdRhsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void setup() override { outInfo(getOutIndex()) = inInfo(getGradInIndex()); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<BinaryLhsGradOp>(*this);
  }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {getGradInIndex(), BinaryOp::getOutIndex(), GradOpInType::GradOut},
        {getFwdRhsInIndex(), BinaryOp::getRhsInIndex(), GradOpInType::In}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {
        {getOutIndex(), BinaryOp::getLhsInIndex()}};
    return outInfo;
  }
};

class BinaryRhsGradOp : public BaseTestOp {
public:
  BinaryRhsGradOp(const BinaryOp &op)
      : BaseTestOp(OperatorIdentifier("TestOps", "BinaryRhsGradOp", 1),
                   op.Op::getSettings()) {}

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdLhsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void setup() override { outInfo(getOutIndex()) = inInfo(getGradInIndex()); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<BinaryRhsGradOp>(*this);
  }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {getGradInIndex(), BinaryOp::getOutIndex(), GradOpInType::GradOut},
        {getFwdLhsInIndex(), BinaryOp::getLhsInIndex(), GradOpInType::In}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {
        {getOutIndex(), BinaryOp::getRhsInIndex()}};
    return outInfo;
  }
};

/**
 * Like BinaryOp, but only differentiable on its lhs. That is, getGradOps
 * will only return a BinaryLhsGradOp, no BinaryRhsGradOp.
 */
class BinaryOpDifferentiableOnLhsOnly : public BinaryOp {
public:
  BinaryOpDifferentiableOnLhsOnly(const Op::Settings &settings)
      : BinaryOp(settings,
                 OperatorIdentifier("TestOps",
                                    "BinaryOpDifferentiableOnLhsOnly",
                                    1)) {}

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<BinaryOpDifferentiableOnLhsOnly>(*this);
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> grads;
    grads.emplace_back(std::make_unique<BinaryLhsGradOp>(*this));
    return grads;
  }
};
/*
 *  - Three inputs to one output.
 *  - Differentiable on all three inputs.
 *  - Gradient calculation of them requires only the output of the op and the
 *    output gradient only.
 *  - A single TertiaryGradOp calculates all three input gradients (so has three
 *    outputs).
 */
class TertiaryOp : public BaseTestOp {
public:
  TertiaryOp(const Op::Settings &settings)
      : BaseTestOp(OperatorIdentifier("TestOps", "TertiaryOp", 1), settings) {}

  static InIndex getArg0InIndex() { return 0; }
  static InIndex getArg1InIndex() { return 1; }
  static InIndex getArg2InIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  void setup() override { outInfo(getOutIndex()) = inInfo(getArg0InIndex()); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<TertiaryOp>(*this);
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> grads;
    grads.emplace_back(std::make_unique<TertiaryGradOp>(*this));
    return grads;
  }
};

class TertiaryGradOp : public BaseTestOp {
public:
  TertiaryGradOp(const TertiaryOp &op)
      : BaseTestOp(OperatorIdentifier("TestOps", "TertiaryGradOp", 1),
                   op.getSettings()) {}

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdOutInIndex() { return 1; }
  static OutIndex getArg0GradOutIndex() { return 0; }
  static OutIndex getArg1GradOutIndex() { return 1; }
  static OutIndex getArg2GradOutIndex() { return 2; }

  void setup() override {
    outInfo(getArg0GradOutIndex()) = inInfo(getGradInIndex());
    outInfo(getArg1GradOutIndex()) = inInfo(getGradInIndex());
    outInfo(getArg2GradOutIndex()) = inInfo(getGradInIndex());
  }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<TertiaryGradOp>(*this);
  }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {getGradInIndex(), TertiaryOp::getOutIndex(), GradOpInType::GradOut},
        {getFwdOutInIndex(), TertiaryOp::getOutIndex(), GradOpInType::Out}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {
        {getArg0GradOutIndex(), TertiaryOp::getArg0InIndex()},
        {getArg1GradOutIndex(), TertiaryOp::getArg1InIndex()},
        {getArg2GradOutIndex(), TertiaryOp::getArg2InIndex()}};
    return outInfo;
  }
};

bool contains(const ExpectedConnections &ecs, const ExpectedConnection &ec) {
  return std::find(ecs.cbegin(), ecs.cend(), ec) != ecs.cend();
}

} // namespace

/**
 * Calling Autodiff::apply with required grads as an empty vector (not nullopt)
 * should throw.
 *
 * Recall "calculate as many grads as possible" behaviour is achieved by passing
 * nullopt, not an empty vector.
 */
BOOST_AUTO_TEST_CASE(TestNonNullEmptyVectorRequiredGradsRaiseseError) {
  const auto ti = TensorInfo{DataType::FLOAT, Shape{2}};

  Ir ir;

  Graph &mg     = ir.getMainGraph();
  TensorId mg_t = "t";
  std::vector<float> mg_t_data(ti.nelms());
  mg.getTensors().addVarInit(mg_t, ti, mg_t_data.data());

  BOOST_REQUIRE_THROW(
      Autodiff{}.apply(ir,
                       mg.id,
                       Autodiff::TensorIds({mg_t}),
                       Autodiff::TensorIds({}), // bad: defined, empty vector
                       {},
                       AutodiffStitchStrategy::SafeAddFwdOutputs),
      popart::error);

  BOOST_REQUIRE_NO_THROW(
      Autodiff{}.apply(ir,
                       mg.id,
                       Autodiff::TensorIds({mg_t}),
                       nonstd::nullopt, // good: undefined vector
                       {},
                       AutodiffStitchStrategy::SafeAddFwdOutputs));

  BOOST_REQUIRE_NO_THROW(
      Autodiff{}.apply(ir,
                       mg.id,
                       nonstd::nullopt, // nullopt here is okay too
                       nonstd::nullopt, // good: undefined vector
                       {},
                       AutodiffStitchStrategy::SafeAddFwdOutputs));
}

/**
 * Tests that calling autodiff with the required grads or provided grads
 * parameters containing tensor ids not in the fwd graph, an error is thrown.
 */
BOOST_AUTO_TEST_CASE(
    TestCallingWithProvidedOrRequiredIdsNotInFwdGraphRaisesError) {

  const auto ti = TensorInfo{DataType::FLOAT, Shape{2}};

  Ir ir;

  Graph &mg     = ir.getMainGraph();
  TensorId mg_t = "t";
  std::vector<float> mg_t_data(ti.nelms());
  mg.getTensors().addVarInit(mg_t, ti, mg_t_data.data());

  // Catches bad id in both gradsProvided and gradsRequired.

  BOOST_REQUIRE_THROW(
      Autodiff{}.apply(ir,
                       mg.id,
                       Autodiff::TensorIds({"bad_name"}),
                       Autodiff::TensorIds({mg_t}),
                       {},
                       AutodiffStitchStrategy::SafeAddFwdOutputs),
      popart::error);

  BOOST_REQUIRE_THROW(
      Autodiff{}.apply(ir,
                       mg.id,
                       Autodiff::TensorIds({mg_t}),
                       Autodiff::TensorIds({"bad_name"}),
                       {},
                       AutodiffStitchStrategy::SafeAddFwdOutputs),
      popart::error);

  Graph &other     = ir.createGraph(GraphId{"other"});
  TensorId other_t = addScope(other, "t");
  other.addInput(other_t, ti);
  other.markAsOutput(other_t);

  // Catches when id is valid but for a different graph.

  BOOST_REQUIRE_THROW(
      Autodiff{}.apply(ir,
                       other.id,
                       Autodiff::TensorIds({other_t}),
                       Autodiff::TensorIds({mg_t}), // bad name
                       {},
                       AutodiffStitchStrategy::SafeAddFwdOutputs),
      popart::error);

  BOOST_REQUIRE_THROW(
      Autodiff{}.apply(ir,
                       other.id,
                       Autodiff::TensorIds({mg_t}), // bad name
                       Autodiff::TensorIds({other_t}),
                       {},
                       AutodiffStitchStrategy::SafeAddFwdOutputs),
      popart::error);

  // Tensors in the fwd graph should not cause an error.

  BOOST_REQUIRE_NO_THROW(
      Autodiff{}.apply(ir,
                       other.id,
                       Autodiff::TensorIds({other_t}),
                       Autodiff::TensorIds({other_t}),
                       {},
                       AutodiffStitchStrategy::SafeAddFwdOutputs));
}

namespace unaryOpTest {

struct SimpleUnaryTestCase {
  GraphId sgId;
  TensorId t0;
  TensorId t1;
  TensorId sg_t0;
  TensorId sg_t1;

  static constexpr VGraphId vgid = 1;
};

SimpleUnaryTestCase initSimpleUnaryTestCase(Ir &ir) {
  // We will set vgid for the fwd ops, then test the bwd ops have the same vgid.
  ir.getSessionOptions().virtualGraphMode = VirtualGraphMode::Manual;
  constexpr VGraphId vgid                 = SimpleUnaryTestCase::vgid;

  Graph &mg = ir.getMainGraph();

  TensorInfo ti{DataType::FLOAT, Shape{2, 2}};
  TensorId t0 = "t0";
  TensorId t1 = "t1";

  mg.createConnectedOp<InitOp>({},
                               {{InitOp::getOutIndex(), t0}},
                               Onnx::CustomOperators::Init_1,
                               ti,
                               TensorType::ActGrad,
                               InitType::Zero,
                               Op::Settings{mg, "Init"})
      ->setVirtualGraphId(vgid);

  Graph &sg = ir.createGraph(GraphId{"sg"});

  auto sg_t0 = addScope(sg, t0);
  auto sg_t1 = addScope(sg, t1);

  sg.addInput(sg_t0, ti);

  sg.createConnectedOp<UnaryOp>({{UnaryOp::getInIndex(), sg_t0}},
                                {{UnaryOp::getOutIndex(), sg_t1}},
                                Op::Settings{sg, "Unary"})
      ->setVirtualGraphId(vgid);

  sg.markAsOutput(sg_t1);

  mg.createConnectedOp<CallOp>({{sg.getInputIndex(sg_t0), t0}},
                               {{sg.getOutputIndex(sg_t1), t1}},
                               Onnx::CustomOperators::Call_1,
                               sg,
                               Op::Settings{mg, "Call"})
      ->setVirtualGraphId(vgid);

  return SimpleUnaryTestCase{std::move(sg.id),
                             std::move(t0),
                             std::move(t1),
                             std::move(sg_t0),
                             std::move(sg_t1)};
}

} // namespace unaryOpTest

/**
 * t0 -> UnaryOp -> t1
 *
 * With different stitch strategies, and with grad for t0 required, or
 * unspecified required grads (which implies compute as many as possible).
 *
 * Additionally, test that if we set UnaryOp's vgid, the gradsum op for t0' will
 * have the same vgid.
 */
BOOST_AUTO_TEST_CASE(TestSimpleUnaryOp) {
  using namespace unaryOpTest;

  auto test = [](const bool specifyRequiredGrads,
                 const AutodiffStitchStrategy stitchStrat) {
    Ir ir;

    const SimpleUnaryTestCase tc = initSimpleUnaryTestCase(ir);
    auto &t0                     = tc.t0;
    auto &t1                     = tc.t1;
    auto &sg_t0                  = tc.sg_t0;
    auto &sg_t1                  = tc.sg_t1;
    auto &mg                     = ir.getMainGraph();
    auto &sg                     = ir.getGraph(tc.sgId);
    // Recall we will test vgid of bwd ops is same as what we set for fwd ops
    constexpr auto vgid = SimpleUnaryTestCase::vgid;

    /*
      provided = all graph outputs (sg_t1)
      required = specifyRequired ? all graph inputs (sg_t0) : null
    */

    // For this fwd graph, Autodiff should produce the exact same graph
    // regardless of `specifyRequiredGrad`.

    nonstd::optional<Autodiff::TensorIds> requiredGrads =
        specifyRequiredGrads
            ? nonstd::make_optional<Autodiff::TensorIds>({sg_t0})
            : nonstd::nullopt;

    Autodiff ad;
    auto f2bInfo = ad.apply(ir,
                            sg.id,
                            Autodiff::TensorIds({sg_t1}),
                            requiredGrads,
                            FwdGraphToBwdGraphInfo{},
                            stitchStrat);

    using popart::irquery::Require;

    // Helper for testing vgid of an irquery::OpTestWrapper
    auto hasVgid = [](const auto &tw_op, const VGraphId vgid) -> bool {
      return tw_op.unwrap()->hasVirtualGraphId() &&
             tw_op.unwrap()->getVirtualGraphId() == vgid;
    };

    irquery::IrTestWrapper tw_ir{ir};

    /* Test main graph has call to fwd sg */

    auto tw_mg = tw_ir.hasGraph(mg.id, Require::MustBeTrue);

    auto tw_call = tw_mg->ops().hasOp<CallOp>(
        [&](auto &tw_call) -> bool {
          return tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_t0), t0) &&
                 tw_call.outputs().hasIdAtIndex(sg.getOutputIndex(sg_t1), t1) &&
                 hasVgid(tw_call, vgid);
        },
        Require::MustBeTrue);

    /* Test sg has correct inputs, outputs, and UnaryOp inside it. */

    auto tw_sg = tw_ir.hasGraph(sg.id, Require::MustBeTrue);

    tw_sg->inputs().hasExactIds({sg_t0}, Require::MustBeTrue);

    // Note extra outputs may have been added to the fwd graph due to stitching.
    tw_sg->outputs().hasExactIds({sg_t1}, Require::MustBeTrue);

    auto tw_unary = tw_sg->ops().hasOp<UnaryOp>(
        [&](auto &tw_unary) -> bool {
          return tw_unary.inputs().hasIdAtIndex(UnaryOp::getInIndex(), sg_t0) &&
                 tw_unary.outputs().hasIdAtIndex(UnaryOp::getOutIndex(),
                                                 sg_t1) &&
                 hasVgid(tw_unary, vgid);
        },
        Require::MustBeTrue);

    /* Test ExpectedConnections of bwd graph `bg`. */

    const auto &bgInfo = f2bInfo.at(sg.id);

    // Check sg_t1 and sg_t1' are in expectedInputs.
    BOOST_REQUIRE(bgInfo.expectedInputs.size() == 2);
    BOOST_REQUIRE(
        contains(bgInfo.expectedInputs,
                 ExpectedConnection{sg_t1, ExpectedConnectionType::Fwd}));
    BOOST_REQUIRE(
        contains(bgInfo.expectedInputs,
                 ExpectedConnection{sg_t1, ExpectedConnectionType::FwdGrad}));

    // Check t0' in expectedOutputs.
    BOOST_REQUIRE(bgInfo.expectedOutputs.size() == 1);
    BOOST_REQUIRE(
        contains(bgInfo.expectedOutputs,
                 ExpectedConnection{sg_t0, ExpectedConnectionType::FwdGrad}));

    /* Test ir has bwd subgraph */

    auto tw_bg = tw_ir.hasGraph(bgInfo.bwdGraphId, Require::MustBeTrue);

    /* Test bg has correct inputs and outputs. */

    // The forward tensors we need must have been cloned into the backward
    // graph, regardless of whether we did recompute or fwdoutput stitching.
    // If this is the main graph, the bwd graph and fwd graph will be the same,
    // so this is a nop.

    auto bg_t1_grad = fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_t1);
    auto bg_t1      = fwdIdToClonedBwdId(sg, tw_bg->unwrap(), sg_t1);
    auto bg_t0_grad = fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_t0);

    tw_bg->inputs().hasExactIds({bg_t1_grad, bg_t1}, Require::MustBeTrue);
    tw_bg->outputs().hasExactIds({bg_t0_grad}, Require::MustBeTrue);

    /* Test bg has correct UnaryGradOp -> edge -> SumOp -> t0'. */

    auto tw_unary_grad = tw_bg->ops().hasOp<UnaryGradOp>(
        [&](auto &tw_unary_grad) -> bool {
          return tw_unary_grad.inputs().hasIdAtIndex(
                     UnaryGradOp::getGradInIndex(), bg_t1_grad) &&
                 tw_unary_grad.inputs().hasIdAtIndex(
                     UnaryGradOp::getFwdArgInIndex(), bg_t1) &&
                 hasVgid(tw_unary_grad, vgid);
        },
        Require::MustBeTrue);

    auto tw_edge_grad =
        tw_unary_grad->outputs()
            .hasIndex(UnaryGradOp::getOutIndex(), Require::MustBeTrue)
            ->tensor();

    tw_edge_grad.consumers().hasOp<SumOp>(
        [&](auto &tw_sum) -> bool {
          return tw_sum.outputs().hasExactIds({bg_t0_grad}) &&
                 hasVgid(tw_sum, vgid);
        },
        Require::MustBeTrue);
  };

  // For this test, all these cases yield the exact same graph.
  for (auto specifyRequiredGrads : {true, false}) {
    for (auto stitchStrat : {AutodiffStitchStrategy::SafeAddFwdOutputs,
                             AutodiffStitchStrategy::RecomputeMinimal}) {
      BOOST_TEST_MESSAGE("TestSimpleUnaryOp: specifyRequiredGrads="
                         << specifyRequiredGrads << ", " << stitchStrat);
      test(specifyRequiredGrads, stitchStrat);
    }
  }
}

namespace binaryOpTest {

struct SimpleBinaryTestCase {
  GraphId sgId;
  TensorId t0;
  TensorId t1;
  TensorId t2;
  TensorId sg_t0;
  TensorId sg_t1;
  TensorId sg_t2;
};

SimpleBinaryTestCase initSimpleBinaryTestCase(Ir &ir) {
  Graph &mg = ir.getMainGraph();

  TensorInfo ti{DataType::FLOAT, Shape{2, 2}};
  TensorId t0 = "t0";
  TensorId t1 = "t1";
  TensorId t2 = "t2";

  mg.createConnectedOp<InitOp>({},
                               {{InitOp::getOutIndex(), t0}},
                               Onnx::CustomOperators::Init_1,
                               ti,
                               TensorType::ActGrad,
                               InitType::Zero,
                               Op::Settings{mg, "Init"});

  mg.createConnectedOp<InitOp>({},
                               {{InitOp::getOutIndex(), t1}},
                               Onnx::CustomOperators::Init_1,
                               ti,
                               TensorType::ActGrad,
                               InitType::Zero,
                               Op::Settings{mg, "Init"});

  Graph &sg = ir.createGraph(GraphId{"sg"});

  TensorId sg_t0 = addScope(sg, t0);
  TensorId sg_t1 = addScope(sg, t1);
  TensorId sg_t2 = addScope(sg, t2);

  sg.addInput(sg_t0, ti);
  sg.addInput(sg_t1, ti);

  sg.createConnectedOp<BinaryOp>(
      {{BinaryOp::getLhsInIndex(), sg_t0}, {BinaryOp::getRhsInIndex(), sg_t1}},
      {{BinaryOp::getOutIndex(), sg_t2}},
      Op::Settings{sg, "Binary"});

  sg.markAsOutput(sg_t2);

  mg.createConnectedOp<CallOp>(
      {{sg.getInputIndex(sg_t0), t0}, {sg.getInputIndex(sg_t1), t1}},
      {{sg.getOutputIndex(sg_t2), t2}},
      Onnx::CustomOperators::Call_1,
      sg,
      Op::Settings{mg, "Call"});

  return SimpleBinaryTestCase{std::move(sg.id),
                              std::move(t0),
                              std::move(t1),
                              std::move(t2),
                              std::move(sg_t0),
                              std::move(sg_t1),
                              std::move(sg_t2)};
}

} // namespace binaryOpTest

/**
 * t0, t1 -> BinaryOp -> t2
 *
 * Test with different stitch strategies, and when requiring grad for t0 only,
 * t1 only, both, or unspecified (which instructs to create as many grads as
 * possible).
 */
BOOST_AUTO_TEST_CASE(TestSimpleBinaryOp) {
  using namespace binaryOpTest;

  auto test = [](const bool requireGrad_t0,
                 const bool requireGrad_t1,
                 const AutodiffStitchStrategy stitchStrat) {
    Ir ir;

    const SimpleBinaryTestCase tc = initSimpleBinaryTestCase(ir);
    auto &t0                      = tc.t0;
    auto &t1                      = tc.t1;
    auto &t2                      = tc.t2;
    auto &sg_t0                   = tc.sg_t0;
    auto &sg_t1                   = tc.sg_t1;
    auto &sg_t2                   = tc.sg_t2;
    auto &mg                      = ir.getMainGraph();
    auto &sg                      = ir.getGraph(tc.sgId);

    /* Apply Autodiff */

    // Because t0 and t1 are inputs of sg, if we pass `nullopt` as
    // `requiredGradsFor`, then both t0' and t1' should be computed (recall the
    // semantics of null `requiredGradsFor` is to compute as many fwd graph
    // input gradients as possible).

    nonstd::optional<Autodiff::TensorIds> requiredGrads;
    if (requireGrad_t0 || requireGrad_t1) {
      requiredGrads = Autodiff::TensorIds{};
      if (requireGrad_t0) {
        requiredGrads->push_back(sg_t0);
      }
      if (requireGrad_t1) {
        requiredGrads->push_back(sg_t1);
      }
    }

    // We will use these throughout to conditionally test parts of the Ir.
    // Recall, we should expect grad of t0/t1 to be computed if it was
    // specifically requested, or if null was requested (compute as many as
    // possible). We expect it to be possible to compute these gradients (when
    // requested) based on the definition of SimpleBinaryOp and its grad ops.
    const bool expectGrad_t0 = requireGrad_t0 || !requireGrad_t1;
    const bool expectGrad_t1 = requireGrad_t1 || !requireGrad_t0;

    // In this case, Autodiff should produce the exact same graph regardless of
    // `specifyRequiredGrad`.

    Autodiff ad;
    auto f2bInfo = ad.apply(ir,
                            sg.id,
                            Autodiff::TensorIds({sg_t2}),
                            requiredGrads,
                            FwdGraphToBwdGraphInfo{},
                            stitchStrat);

    /* TEST */

    using irquery::Require;

    irquery::IrTestWrapper tw_ir{ir};

    /* Test main graph has call to fwd sg */

    auto tw_mg = tw_ir.hasGraph(mg.id, Require::MustBeTrue);

    auto tw_call = tw_mg->ops().hasOp<CallOp>(
        [&](auto &tw_call) -> bool {
          return tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_t0), t0) &&
                 tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_t1), t1) &&
                 tw_call.outputs().hasIdAtIndex(sg.getOutputIndex(sg_t2), t2);
        },
        Require::MustBeTrue);

    /* Test sg has correct inputs, outputs, and BinaryOp inside it. */

    auto tw_sg = tw_ir.hasGraph(sg.id, Require::MustBeTrue);

    tw_sg->inputs().hasExactIds({sg_t0, sg_t1}, Require::MustBeTrue);
    tw_sg->outputs().hasExactIds({sg_t2}, Require::MustBeTrue);

    auto tw_binary = tw_sg->ops().hasOp<BinaryOp>(
        [&](auto &tw_binary) -> bool {
          return tw_binary.inputs().hasIdAtIndex(BinaryOp::getLhsInIndex(),
                                                 sg_t0) &&
                 tw_binary.inputs().hasIdAtIndex(BinaryOp::getRhsInIndex(),
                                                 sg_t1) &&
                 tw_binary.outputs().hasIdAtIndex(BinaryOp::getOutIndex(),
                                                  sg_t2);
        },
        Require::MustBeTrue);

    /* Test ExpectedConnections of bwd graph `bg`. */

    const auto &bgInfo = f2bInfo.at(sg.id);

    // Check t2', t0, and t1 are in expectedInputs.

    const std::size_t numExpectedExpectedInputs =
        1 + (expectGrad_t0 ? 1 : 0) + (expectGrad_t1 ? 1 : 0);

    BOOST_REQUIRE_EQUAL(bgInfo.expectedInputs.size(),
                        numExpectedExpectedInputs);
    BOOST_REQUIRE(
        contains(bgInfo.expectedInputs,
                 ExpectedConnection{sg_t2, ExpectedConnectionType::FwdGrad}));
    if (expectGrad_t1) {
      BOOST_REQUIRE(
          contains(bgInfo.expectedInputs,
                   ExpectedConnection{sg_t0, ExpectedConnectionType::Fwd}));
    }
    if (expectGrad_t0) {
      BOOST_REQUIRE(
          contains(bgInfo.expectedInputs,
                   ExpectedConnection{sg_t1, ExpectedConnectionType::Fwd}));
    }

    // Check t0', t1' in expectedOutputs.

    const std::size_t numExpectedExpectedOutputs =
        (expectGrad_t0 ? 1 : 0) + (expectGrad_t1 ? 1 : 0);

    BOOST_REQUIRE_EQUAL(bgInfo.expectedOutputs.size(),
                        numExpectedExpectedOutputs);
    if (expectGrad_t0) {
      BOOST_REQUIRE(
          contains(bgInfo.expectedOutputs,
                   ExpectedConnection{sg_t0, ExpectedConnectionType::FwdGrad}));
    }
    if (expectGrad_t1) {
      BOOST_REQUIRE(
          contains(bgInfo.expectedOutputs,
                   ExpectedConnection{sg_t1, ExpectedConnectionType::FwdGrad}));
    }
    auto tw_bg = tw_ir.hasGraph(bgInfo.bwdGraphId, Require::MustBeTrue);

    auto bg_t2_grad = fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_t2);
    auto bg_t0 =
        expectGrad_t1 ? fwdIdToClonedBwdId(sg, tw_bg->unwrap(), sg_t0) : "";
    auto bg_t1 =
        expectGrad_t0 ? fwdIdToClonedBwdId(sg, tw_bg->unwrap(), sg_t1) : "";
    auto bg_t0_grad =
        expectGrad_t0 ? fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_t0) : "";
    auto bg_t1_grad =
        expectGrad_t1 ? fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_t1) : "";

    /* Test bg has correct input and outputs */

    std::vector<TensorId> expectedBgInputs = {bg_t2_grad};
    if (expectGrad_t0) {
      expectedBgInputs.push_back(bg_t1);
    }
    if (expectGrad_t1) {
      expectedBgInputs.push_back(bg_t0);
    }
    tw_bg->inputs().hasExactIds(expectedBgInputs, Require::MustBeTrue);

    std::vector<TensorId> expectedBgOutputs = {};
    if (expectGrad_t0) {
      expectedBgOutputs.push_back(bg_t0_grad);
    }
    if (expectGrad_t1) {
      expectedBgOutputs.push_back(bg_t1_grad);
    }
    tw_bg->outputs().hasExactIds(expectedBgOutputs, Require::MustBeTrue);

    /* Test bg has correct BinaryLhsGradOp -> SumOp -> bg_t0_grad */

    if (expectGrad_t0) {
      auto tw_binary_lhs_grad = tw_bg->ops().hasOp<BinaryLhsGradOp>(
          [&](auto &tw_binary_lhs_grad) -> bool {
            return tw_binary_lhs_grad.inputs().hasIdAtIndex(
                       BinaryLhsGradOp::getFwdRhsInIndex(), bg_t1) &&
                   tw_binary_lhs_grad.inputs().hasIdAtIndex(
                       BinaryLhsGradOp::getGradInIndex(), bg_t2_grad);
          },
          Require::MustBeTrue);

      auto tw_edge_grad_t0 =
          tw_binary_lhs_grad->outputs()
              .hasIndex(BinaryLhsGradOp::getOutIndex(), Require::MustBeTrue)
              ->tensor();

      tw_edge_grad_t0.consumers().hasOp<SumOp>(
          [&](auto &tw_sum) -> bool {
            return tw_sum.outputs().hasExactIds({bg_t0_grad});
          },
          Require::MustBeTrue);
    } else {
      tw_bg->ops().hasOp<BinaryLhsGradOp>(Require::MustBeFalse);
    }

    /* Test bg has correct BinaryRhsGradOp -> SumOp -> bg_t1_grad */

    if (expectGrad_t1) {
      auto tw_binary_rhs_grad = tw_bg->ops().hasOp<BinaryRhsGradOp>(
          [&](auto &tw_binary_rhs_grad) -> bool {
            return tw_binary_rhs_grad.inputs().hasIdAtIndex(
                       BinaryRhsGradOp::getFwdLhsInIndex(), bg_t0) &&
                   tw_binary_rhs_grad.inputs().hasIdAtIndex(
                       BinaryRhsGradOp::getGradInIndex(), bg_t2_grad);
          },
          Require::MustBeTrue);

      auto tw_edge_grad_t1 =
          tw_binary_rhs_grad->outputs()
              .hasIndex(BinaryRhsGradOp::getOutIndex(), Require::MustBeTrue)
              ->tensor();

      tw_edge_grad_t1.consumers().hasOp<SumOp>(
          [&](auto &tw_sum) -> bool {
            return tw_sum.outputs().hasExactIds({bg_t1_grad});
          },
          Require::MustBeTrue);
    } else {
      tw_bg->ops().hasOp<BinaryRhsGradOp>(Require::MustBeFalse);
    }
  }; // lambda test

  // Note: For this test, all stitch strategies yield the same graph. This is
  // because all fwd graph tensors required in the backward graph are always
  // available (from the main graph) - there are no intermediate tensors that
  // only exist in the subgraph that need to be used in the backward graph.
  //
  // Note: (false, false) corresponds to requiring `nullopt`, which means all
  // possible input grads, so both t0 and t1.
  for (auto requireGrad_t0 : {true, false}) {
    for (auto requireGrad_t1 : {true, false}) {
      for (auto stitchStrat : {AutodiffStitchStrategy::SafeAddFwdOutputs,
                               AutodiffStitchStrategy::RecomputeMinimal}) {
        BOOST_TEST_MESSAGE("TestSimpleBinaryOp: requireGrad_t0="
                           << requireGrad_t0 << ", requireGrad_t1="
                           << requireGrad_t1 << ", " << stitchStrat);
        test(requireGrad_t0, requireGrad_t1, stitchStrat);
      }
    }
  }
}

/**
 *   --------------------
 * -|-> a -> Unary -> c -|->
 * -|-> b -> Unary -> d -|->
 *   --------------------
 *
 * Only require a', only provide c'. The resulting backward graph should be:
 *
 *    --------------------------
 *   |                  /- c  <-|-
 * <-|- a' <- UnaryGrad <- c' <-|-
 *    --------------------------
 *
 * That is, Autodiff should be able to handle only needing the minimal
 * "provided" tensors to compute the "required" grads. No unnecessary ops,
 * such as a UnaryGrad for b, should be in the graph.
 */
BOOST_AUTO_TEST_CASE(
    TestTwoDisconnectedPathsInSubgraphAndOnlyDifferentiateAlongOne) {
  /**** BUILD FWD GRAPH ******/

  Ir ir;

  Graph &mg = ir.getMainGraph();
  Graph &sg = ir.createGraph(GraphId{"sg"});

  TensorInfo ti{DataType::FLOAT, Shape{2, 2}};

  TensorId a = "a";
  TensorId b = "b";
  TensorId c = "c";
  TensorId d = "d";

  // The input/output maps we will pass when creating the CallOp in main graph.
  std::map<InIndex, TensorId> callOpIns;
  std::map<OutIndex, TensorId> callOpOuts;

  std::vector<std::tuple<TensorId &, TensorId &>> paths{{a, c}, {b, d}};
  for (const auto &path : paths) {
    auto &in  = std::get<0>(path);
    auto &out = std::get<1>(path);

    mg.createConnectedOp<InitOp>({},
                                 {{InitOp::getOutIndex(), in}},
                                 Onnx::CustomOperators::Init_1,
                                 ti,
                                 TensorType::ActGrad,
                                 InitType::Zero,
                                 Op::Settings{mg, "Init-" + in});

    auto sg_in  = addScope(sg, in);
    auto sg_out = addScope(sg, out);

    sg.addInput(sg_in, ti);
    callOpIns.insert({sg.getInputIndex(sg_in), in});

    sg.createConnectedOp<UnaryOp>({{UnaryOp::getInIndex(), sg_in}},
                                  {{UnaryOp::getOutIndex(), sg_out}},
                                  Op::Settings{sg, "Unary-" + sg_in});

    sg.markAsOutput(sg_out);
    callOpOuts.insert({sg.getOutputIndex(sg_out), out});
  }

  mg.createConnectedOp<CallOp>(callOpIns,
                               callOpOuts,
                               Onnx::CustomOperators::Call_1,
                               sg,
                               Op::Settings{mg, "Call"});

  /****** APPLY AUTODIFF ******/

  auto sg_a = addScope(sg, a);
  auto sg_b = addScope(sg, b);
  auto sg_c = addScope(sg, c);
  auto sg_d = addScope(sg, d);

  Autodiff ad;
  auto f2bInfo = ad.apply(ir,
                          sg.id,
                          Autodiff::TensorIds({sg_c}), // only provide c
                          Autodiff::TensorIds({sg_a}), // only require a
                          FwdGraphToBwdGraphInfo{},
                          AutodiffStitchStrategy::SafeAddFwdOutputs);

  /****** TEST ******/

  using irquery::Require;

  irquery::IrTestWrapper tw_ir{ir};

  auto tw_mg = tw_ir.hasGraph(mg.id, Require::MustBeTrue);

  auto tw_call = tw_mg->ops().hasOp<CallOp>(
      [&](auto &tw_call) -> bool {
        return tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_a), a) &&
               tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_b), b) &&
               tw_call.outputs().hasIdAtIndex(sg.getOutputIndex(sg_c), c) &&
               tw_call.outputs().hasIdAtIndex(sg.getOutputIndex(sg_d), d);
      },
      Require::MustBeTrue);

  /* Test sg has correct inputs, outputs */

  auto tw_sg = tw_ir.hasGraph(sg.id, Require::MustBeTrue);

  tw_sg->inputs().hasExactIds({sg_a, sg_b}, Require::MustBeTrue);

  // Note extra outputs may have been added to the fwd graph due to stitching.
  tw_sg->outputs().hasExactIds({sg_c, sg_d}, Require::MustBeTrue);

  /* Test a -> Unary -> c */

  auto tw_unary_a = tw_sg->ops().hasOp<UnaryOp>(
      [&](auto &tw_unary) -> bool {
        return tw_unary.inputs().hasIdAtIndex(UnaryOp::getInIndex(), sg_a) &&
               tw_unary.outputs().hasIdAtIndex(UnaryOp::getOutIndex(), sg_c);
      },
      Require::MustBeTrue);

  /* Test b -> Unary -> d */

  auto tw_unary_b = tw_sg->ops().hasOp<UnaryOp>(
      [&](auto &tw_unary) -> bool {
        return tw_unary.inputs().hasIdAtIndex(UnaryOp::getInIndex(), sg_b) &&
               tw_unary.outputs().hasIdAtIndex(UnaryOp::getOutIndex(), sg_d);
      },
      Require::MustBeTrue);

  /* Test ExpectedConnections of bwd graph `bg`. */

  const auto &bgInfo = f2bInfo.at(sg.id);

  // Check c and c' are in expectedInputs.
  BOOST_REQUIRE(bgInfo.expectedInputs.size() == 2);
  BOOST_REQUIRE(
      contains(bgInfo.expectedInputs,
               ExpectedConnection{sg_c, ExpectedConnectionType::Fwd}));
  BOOST_REQUIRE(
      contains(bgInfo.expectedInputs,
               ExpectedConnection{sg_c, ExpectedConnectionType::FwdGrad}));

  // Check a' in expectedOutputs.
  BOOST_REQUIRE(bgInfo.expectedOutputs.size() == 1);
  BOOST_REQUIRE(
      contains(bgInfo.expectedOutputs,
               ExpectedConnection{sg_a, ExpectedConnectionType::FwdGrad}));

  /* Test ir has bwd subgraph */

  auto tw_bg = tw_ir.hasGraph(bgInfo.bwdGraphId, Require::MustBeTrue);

  /* Test bg has correct inputs and outputs. */

  // The forward tensors we need must have been cloned into the backward
  // graph, regardless of whether we did recompute or fwdoutput stitching.
  // If this is the main graph, the bwd graph and fwd graph will be the same,
  // so this is a nop.

  auto bg_c_grad = fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_c);
  auto bg_c      = fwdIdToClonedBwdId(sg, tw_bg->unwrap(), sg_c);
  auto bg_a_grad = fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_a);

  tw_bg->inputs().hasExactIds({bg_c_grad, bg_c}, Require::MustBeTrue);
  tw_bg->outputs().hasExactIds({bg_a_grad}, Require::MustBeTrue);

  /* Test bg has correct UnaryGradOp -> edge -> SumOp -> a'. */

  auto tw_unary_grad = tw_bg->ops().hasOp<UnaryGradOp>(
      [&](auto &tw_unary_grad) -> bool {
        return tw_unary_grad.inputs().hasIdAtIndex(
                   UnaryGradOp::getGradInIndex(), bg_c_grad) &&
               tw_unary_grad.inputs().hasIdAtIndex(
                   UnaryGradOp::getFwdArgInIndex(), bg_c);
      },
      Require::MustBeTrue);

  auto tw_edge_grad =
      tw_unary_grad->outputs()
          .hasIndex(UnaryGradOp::getOutIndex(), Require::MustBeTrue)
          ->tensor();

  tw_edge_grad.consumers().hasOp<SumOp>(
      [&](auto &tw_sum) -> bool {
        return tw_sum.outputs().hasExactIds({bg_a_grad});
      },
      Require::MustBeTrue);

  /* Test bg does not contain a, b or d. */

  const auto &bg_tensors = tw_bg->unwrap().get().getTensors();
  BOOST_REQUIRE(!bg_tensors.contains(a));
  BOOST_REQUIRE(!bg_tensors.contains(b));
  BOOST_REQUIRE(!bg_tensors.contains(d));
}

namespace nonDifferentiableTest {

struct TestCase {
  std::unique_ptr<Ir> ir;
  GraphId sgId;
  TensorId a;
  TensorId b;
  TensorId c;
  TensorId sg_a;
  TensorId sg_b;
  TensorId sg_t;
  TensorId sg_c;
};

// Only require b, or request all possible (but require none).
template <bool SpecifyRequiredGrads>
nonstd::optional<Autodiff::TensorIds> requiredGrads(const TestCase &);
template <>
nonstd::optional<Autodiff::TensorIds> requiredGrads<true>(const TestCase &tc) {
  return Autodiff::TensorIds({tc.sg_b});
}
template <>
nonstd::optional<Autodiff::TensorIds> requiredGrads<false>(const TestCase &) {
  return nonstd::nullopt;
}

template <AutodiffStitchStrategy S>
std::vector<TensorId> expectedSgOutputs(const TestCase &);
template <>
std::vector<TensorId>
expectedSgOutputs<AutodiffStitchStrategy::SafeAddFwdOutputs>(
    const TestCase &tc) {
  // t will be taken from the fwd graph and input into the bwd graph.
  return {tc.sg_c, tc.sg_t};
}
template <>
std::vector<TensorId>
expectedSgOutputs<AutodiffStitchStrategy::RecomputeMinimal>(
    const TestCase &tc) {
  // t will be recomputed from a. However, as a is an input, it will not
  // be output from sg and instead the user will be expected to directly
  // thread it through from the parent of sg.
  return {tc.sg_c};
}

template <AutodiffStitchStrategy S>
std::vector<ExpectedConnection> expectedExpectedInputs(const TestCase &);
template <>
std::vector<ExpectedConnection>
expectedExpectedInputs<AutodiffStitchStrategy::SafeAddFwdOutputs>(
    const TestCase &tc) {
  // c' to seed the grad graph, and t for the BinaryRhsGradOp.
  // t will be passed in as input from the fwd graph.
  return {{tc.sg_c, ExpectedConnectionType::FwdGrad},
          {tc.sg_t, ExpectedConnectionType::Fwd}};
}
template <>
std::vector<ExpectedConnection>
expectedExpectedInputs<AutodiffStitchStrategy::RecomputeMinimal>(
    const TestCase &tc) {
  // c' to seed the grad graph, and t for the BinaryRhsGradOp.
  // However t will be recomputed in the backward graph from a, so a is an input
  // and not t.
  return {{tc.sg_c, ExpectedConnectionType::FwdGrad},
          {tc.sg_a, ExpectedConnectionType::Fwd}};
}

struct TestCaseBgTensors {
  TensorId bg_a;
  TensorId bg_t;
  TensorId bg_c_grad;
  TensorId bg_b_grad;
};

template <AutodiffStitchStrategy S>
TestCaseBgTensors bgTensors(const Graph &sg, const Graph &bg, const TestCase &);
template <>
TestCaseBgTensors
bgTensors<AutodiffStitchStrategy::SafeAddFwdOutputs>(const Graph &sg,
                                                     const Graph &bg,
                                                     const TestCase &tc) {
  return {"",
          fwdIdToClonedBwdId(sg, bg, tc.sg_t),
          fwdIdToBwdGradId(sg, bg, tc.sg_c),
          fwdIdToBwdGradId(sg, bg, tc.sg_b)};
}
template <>
TestCaseBgTensors
bgTensors<AutodiffStitchStrategy::RecomputeMinimal>(const Graph &sg,
                                                    const Graph &bg,
                                                    const TestCase &tc) {
  return {fwdIdToClonedBwdId(sg, bg, tc.sg_a),
          fwdIdToClonedBwdId(sg, bg, tc.sg_t),
          fwdIdToBwdGradId(sg, bg, tc.sg_c),
          fwdIdToBwdGradId(sg, bg, tc.sg_b)};
}

template <AutodiffStitchStrategy S>
std::vector<TensorId> expectedBgInputs(const TestCaseBgTensors &);
template <>
std::vector<TensorId>
expectedBgInputs<AutodiffStitchStrategy::SafeAddFwdOutputs>(
    const TestCaseBgTensors &tc) {
  // t will be passed in.
  return {tc.bg_c_grad, tc.bg_t};
}
template <>
std::vector<TensorId>
expectedBgInputs<AutodiffStitchStrategy::RecomputeMinimal>(
    const TestCaseBgTensors &tc) {
  // t will be recomputed from a.
  return {tc.bg_c_grad, tc.bg_a};
}

template <AutodiffStitchStrategy S, typename... Outputs>
struct RecomputedUnaryTestCriteria {
  std::tuple<Outputs...> operator()(const TestCaseBgTensors &);
};
// There should be no recomputed unary op.
template <>
struct RecomputedUnaryTestCriteria<AutodiffStitchStrategy::SafeAddFwdOutputs> {
  std::tuple<irquery::Require> operator()(const TestCaseBgTensors &) {
    return std::make_tuple(irquery::Require::MustBeFalse);
  }
};
// There should be a recomputed UnaryOp connected as follows.
template <>
struct RecomputedUnaryTestCriteria<AutodiffStitchStrategy::RecomputeMinimal> {
  std::tuple<irquery::OpsTestWrapper::OpPred<UnaryOp>, irquery::Require>
  operator()(const TestCaseBgTensors &tc) {
    return std::make_tuple(
        [&](auto &tw_unary) {
          return tw_unary.inputs().hasIdAtIndex(UnaryOp::getInIndex(),
                                                tc.bg_a) &&
                 tw_unary.outputs().hasIdAtIndex(UnaryOp::getOutIndex(),
                                                 tc.bg_t);
        },
        irquery::Require::MustBeTrue);
  }
};

TestCase initTestCase() {
  /**** BUILD FWD GRAPH ******/

  auto ir = std::make_unique<Ir>();

  Graph &mg = ir->getMainGraph();

  TensorInfo ti{DataType::FLOAT, Shape{2, 2}};

  TensorId a = "a";
  TensorId b = "b";
  TensorId c = "c";

  // Just to mix things up, make `a` a Variable.
  std::vector<float> a_host(ti.nelms());
  mg.getTensors().addVarInit(a, ti, a_host.data(), "a");

  mg.createConnectedOp<InitOp>({},
                               {{InitOp::getOutIndex(), b}},
                               Onnx::CustomOperators::Init_1,
                               ti,
                               TensorType::ActGrad,
                               InitType::Zero,
                               Op::Settings{mg, "Init"});

  Graph &sg = ir->createGraph(GraphId{"sg"});

  auto sg_a = addScope(sg, a);
  auto sg_b = addScope(sg, b);
  auto sg_c = addScope(sg, c);
  auto sg_t = addScope(sg, TensorId{"t"});

  sg.addInput(sg_a, ti);
  sg.addInput(sg_b, ti);

  sg.createConnectedOp<UnaryOp>({{UnaryOp::getInIndex(), sg_a}},
                                {{UnaryOp::getOutIndex(), sg_t}},
                                Op::Settings{sg, "Unary"});

  sg.createConnectedOp<BinaryOpDifferentiableOnLhsOnly>(
      {{BinaryOp::getLhsInIndex(), sg_b}, {BinaryOp::getRhsInIndex(), sg_t}},
      {{BinaryOp::getOutIndex(), sg_c}},
      Op::Settings{sg, "BinaryOpDifferentiableOnLhsOnly"});

  sg.markAsOutput(sg_c);

  mg.createConnectedOp<CallOp>(
      {{sg.getInputIndex(sg_a), a}, {sg.getInputIndex(sg_b), b}},
      {{sg.getOutputIndex(sg_c), c}},
      Onnx::CustomOperators::Call_1,
      sg,
      Op::Settings{mg, "Call"});

  return TestCase{std::move(ir),
                  std::move(sg.id),
                  std::move(a),
                  std::move(b),
                  std::move(c),
                  std::move(sg_a),
                  std::move(sg_b),
                  std::move(sg_t),
                  std::move(sg_c)};
}

template <bool SpecifyRequiredGrads, AutodiffStitchStrategy StitchStrat>
void test() {
  const auto tc = initTestCase();
  auto &ir      = *tc.ir;
  auto &sgId    = tc.sgId;
  auto &a       = tc.a;
  auto &b       = tc.b;
  auto &c       = tc.c;
  auto &sg_a    = tc.sg_a;
  auto &sg_b    = tc.sg_b;
  auto &sg_t    = tc.sg_t;
  auto &sg_c    = tc.sg_c;
  auto &mg      = ir.getMainGraph();
  auto &sg      = ir.getGraph(sgId);

  /****** APPLY AUTODIFF ******/

  Autodiff ad;
  auto f2bInfo = ad.apply(ir,
                          sg.id,
                          Autodiff::TensorIds({sg_c}), // only provide c
                          requiredGrads<SpecifyRequiredGrads>(tc),
                          FwdGraphToBwdGraphInfo{},
                          StitchStrat);

  /****** TEST ******/

  using irquery::Require;

  irquery::IrTestWrapper tw_ir{ir};

  /* Test main graph exists and has CallOp */

  auto tw_mg = tw_ir.hasGraph(mg.id, Require::MustBeTrue);

  auto tw_call = tw_mg->ops().template hasOp<CallOp>(
      [&](auto &tw_call) -> bool {
        return tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_a), a) &&
               tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_b), b) &&
               tw_call.outputs().hasIdAtIndex(sg.getOutputIndex(sg_c), c);
      },
      Require::MustBeTrue);

  /* Test sg has correct inputs, outputs */

  auto tw_sg = tw_ir.hasGraph(sg.id, Require::MustBeTrue);

  tw_sg->inputs().hasExactIds({sg_a, sg_b}, Require::MustBeTrue);
  tw_sg->outputs().hasExactIds(expectedSgOutputs<StitchStrat>(tc),
                               Require::MustBeTrue);

  /* Test a -> Unary -> t */

  auto tw_unary = tw_sg->ops().template hasOp<UnaryOp>(
      [&](auto &tw_unary) -> bool {
        return tw_unary.inputs().hasIdAtIndex(UnaryOp::getInIndex(), sg_a) &&
               tw_unary.outputs().hasIdAtIndex(UnaryOp::getOutIndex(), sg_t);
      },
      Require::MustBeTrue);

  auto tw_binary = tw_sg->ops().template hasOp<BinaryOpDifferentiableOnLhsOnly>(
      [&](auto &tw_binary) -> bool {
        return tw_binary.inputs().hasIdAtIndex(BinaryOp::getLhsInIndex(),
                                               sg_b) &&
               tw_binary.inputs().hasIdAtIndex(BinaryOp::getRhsInIndex(),
                                               sg_t) &&
               tw_binary.outputs().hasIdAtIndex(BinaryOp::getOutIndex(), sg_c);
      },
      Require::MustBeTrue);

  /* Test ExpectedConnections of bwd graph `bg`. */

  const auto &bgInfo = f2bInfo.at(sg.id);

  const auto expectedExpectedIns = expectedExpectedInputs<StitchStrat>(tc);
  BOOST_REQUIRE_EQUAL(bgInfo.expectedInputs.size(), expectedExpectedIns.size());

  for (const auto &ec : expectedExpectedIns) {
    BOOST_REQUIRE(contains(bgInfo.expectedInputs, ec));
  }
  // Check b' in expectedOutputs.
  BOOST_REQUIRE_EQUAL(bgInfo.expectedOutputs.size(), 1);
  BOOST_REQUIRE(
      contains(bgInfo.expectedOutputs,
               ExpectedConnection{sg_b, ExpectedConnectionType::FwdGrad}));

  /* Test ir has bwd subgraph */

  auto tw_bg = tw_ir.hasGraph(bgInfo.bwdGraphId, Require::MustBeTrue);

  /* Test bg has correct inputs and outputs. */

  // The forward tensors we need must have been cloned into the backward
  // graph, regardless of whether we did recompute or fwdoutput stitching.
  // If this is the main graph, the bwd graph and fwd graph will be the same,
  // so this is a nop.

  const auto &tcbg = bgTensors<StitchStrat>(sg, tw_bg->unwrap(), tc);
  auto &bg_a       = tcbg.bg_a;
  auto &bg_t       = tcbg.bg_t;
  auto &bg_c_grad  = tcbg.bg_c_grad;
  auto &bg_b_grad  = tcbg.bg_b_grad;
  (void)bg_a;

  tw_bg->inputs().hasExactIds(expectedBgInputs<StitchStrat>(tcbg),
                              Require::MustBeTrue);
  tw_bg->outputs().hasExactIds({bg_b_grad}, Require::MustBeTrue);

  /* Test bg has correct BinaryLhsGradOp -> SumOp -> bg_b_grad */

  auto tw_binary_lhs_grad = tw_bg->ops().template hasOp<BinaryLhsGradOp>(
      [&](auto &tw_binary_lhs_grad) -> bool {
        return tw_binary_lhs_grad.inputs().hasIdAtIndex(
                   BinaryLhsGradOp::getFwdRhsInIndex(), bg_t) &&
               tw_binary_lhs_grad.inputs().hasIdAtIndex(
                   BinaryLhsGradOp::getGradInIndex(), bg_c_grad);
      },
      Require::MustBeTrue);

  auto tw_edge_grad =
      tw_binary_lhs_grad->outputs()
          .hasIndex(BinaryLhsGradOp::getOutIndex(), Require::MustBeTrue)
          ->tensor();

  tw_edge_grad.consumers().template hasOp<SumOp>(
      [&](auto &tw_sum) -> bool {
        return tw_sum.outputs().hasExactIds({bg_b_grad});
      },
      Require::MustBeTrue);

  /* Test no grad ops for rhs or unary. */

  tw_bg->ops().template hasOp<BinaryRhsGradOp>(Require::MustBeFalse);
  tw_bg->ops().template hasOp<UnaryGradOp>(Require::MustBeFalse);

  /* If recompute stitching, test t correctly recomputed from a. */

  // Generic lambda that binds tw_bg->ops().hasOp<UnaryOp> to variadic
  // parameters.
  auto hasUnaryOpFn = [&tw_bg](auto &&args...) {
    return tw_bg->ops().template hasOp<UnaryOp>(
        std::forward<decltype(args)>(args));
  };

  // Apply tw_bg->ops().hasOp<UnaryOp> to the parameters required for this
  // stitch strategy. These are the `Require` and the OpPred, if there is one.
  boost::hof::unpack(hasUnaryOpFn)(
      RecomputedUnaryTestCriteria<StitchStrat>{}(tcbg));
}

} // namespace nonDifferentiableTest

/**
 * a -> A -> t -> C -> c
 *           b -/
 *
 * Where:
 *   - C is not differentiable on the input that is t.
 *   - C is differentiable on the input that is b.
 *   - Only grad for b is required, not t or a.
 *
 * Autodiff should not fail on attempting to find t'.
 * Autodiff should not fail on attempting to create A' but t' does not exist.
 */
BOOST_AUTO_TEST_CASE(TestCanHandleNonDifferentiableInput) {
  using namespace nonDifferentiableTest;

  test<false, AutodiffStitchStrategy::SafeAddFwdOutputs>();
  test<false, AutodiffStitchStrategy::RecomputeMinimal>();
  test<true, AutodiffStitchStrategy::SafeAddFwdOutputs>();
  test<true, AutodiffStitchStrategy::RecomputeMinimal>();
}

/**
 * a, b, c -> TertiaryOp -> d
 *
 * Where:
 *   - TertiaryOp has one TertiaryGradOp that produces grads for a, b, c.
 *   - Require grads for a, b only.
 *
 * Should result in backward graph:
 *
 * a', b', c' <- TertiaryGradOp <- d, d'
 *
 * Where:
 *   - Only a', b' are outputs.
 *   - c' has not been pruned, as otherwise TertiaryGradOp is invalid.
 *
 * This is because, in PopART Ir, if an Op exists then all of its output tensors
 * must exist too. Breaking this assumption causes an error in the Opx when
 * lowering the Op.
 */
// TODO(T55417): Enable this test. The bug is known but not fixed:
//     BackwardsGraphCreatorHelper::doPrune(graph) will eagerly prune a tensor
//     regardless of whether its producer op is going to be pruned or not.
BOOST_AUTO_TEST_CASE(
    TestDoesNotPruneUnrequiredGradTensorIfProducingGradOpStillRequired,
    *boost::unit_test::disabled()) {
  auto test = [](AutodiffStitchStrategy stitchStrat) {
    Ir ir;

    /* Build fwd graph */

    Graph &mg = ir.getMainGraph();

    TensorInfo ti{DataType::FLOAT, Shape{2, 2}};
    TensorId a = "a";
    TensorId b = "b";
    TensorId c = "c";
    TensorId d = "d";

    for (const auto &tid : {a, b, c}) {
      mg.createConnectedOp<InitOp>({},
                                   {{InitOp::getOutIndex(), tid}},
                                   Onnx::CustomOperators::Init_1,
                                   ti,
                                   TensorType::ActGrad,
                                   InitType::Zero,
                                   Op::Settings{mg, "Init-" + tid});
    }

    Graph &sg = ir.createGraph(GraphId{"sg"});

    auto sg_a = addScope(sg, a);
    auto sg_b = addScope(sg, b);
    auto sg_c = addScope(sg, c);
    auto sg_d = addScope(sg, d);

    sg.addInput(sg_a, ti);
    sg.addInput(sg_c, ti);
    sg.addInput(sg_b, ti);

    sg.createConnectedOp<TertiaryOp>(
        {
            {TertiaryOp::getArg0InIndex(), sg_a},
            {TertiaryOp::getArg1InIndex(), sg_b},
            {TertiaryOp::getArg2InIndex(), sg_c},

        },
        {{UnaryOp::getOutIndex(), sg_d}},

        Op::Settings{sg, "Tertiary"});

    sg.markAsOutput(sg_d);

    mg.createConnectedOp<CallOp>(
        {
            {sg.getInputIndex(sg_a), a},
            {sg.getInputIndex(sg_b), b},
            {sg.getInputIndex(sg_c), c},

        },
        {{sg.getOutputIndex(sg_d), d}},
        Onnx::CustomOperators::Call_1,
        sg,
        Op::Settings{mg, "Call"});

    /* Apply Autodiff */

    Autodiff ad;
    auto f2bInfo = ad.apply(ir,
                            sg.id,
                            Autodiff::TensorIds({sg_d}),
                            Autodiff::TensorIds({sg_a, sg_b}),
                            FwdGraphToBwdGraphInfo{},
                            stitchStrat);
    /****** TEST ******/

    using irquery::Require;

    irquery::IrTestWrapper tw_ir{ir};

    auto tw_mg = tw_ir.hasGraph(mg.id, Require::MustBeTrue);

    auto tw_call = tw_mg->ops().hasOp<CallOp>(
        [&](auto &tw_call) -> bool {
          return tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_a), a) &&
                 tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_b), b) &&
                 tw_call.inputs().hasIdAtIndex(sg.getInputIndex(sg_c), c) &&
                 tw_call.outputs().hasIdAtIndex(sg.getOutputIndex(sg_d), d);
        },
        Require::MustBeTrue);

    /* Test sg has correct inputs, outputs */

    auto tw_sg = tw_ir.hasGraph(sg.id, Require::MustBeTrue);

    tw_sg->inputs().hasExactIds({sg_a, sg_b, sg_c}, Require::MustBeTrue);

    // Regardless of stitching strategy, the fwd graph will not need any
    // additional outputs in this case.
    tw_sg->outputs().hasExactIds({sg_d}, Require::MustBeTrue);

    /* Test a, b, c -> Unary -> d */

    tw_sg->ops().hasOp<TertiaryOp>(
        [&](auto &tw_tertiary) -> bool {
          return tw_tertiary.inputs().hasIdAtIndex(TertiaryOp::getArg0InIndex(),
                                                   sg_a) &&
                 tw_tertiary.inputs().hasIdAtIndex(TertiaryOp::getArg1InIndex(),
                                                   sg_b) &&
                 tw_tertiary.inputs().hasIdAtIndex(TertiaryOp::getArg2InIndex(),
                                                   sg_c) &&
                 tw_tertiary.outputs().hasIdAtIndex(TertiaryOp::getOutIndex(),
                                                    sg_d);
        },
        Require::MustBeTrue);

    /* Test ExpectedConnections of bwd graph `bg`. */

    const auto &bgInfo = f2bInfo.at(sg.id);

    // Check d and d' are in expectedInputs.
    BOOST_REQUIRE(bgInfo.expectedInputs.size() == 2);
    BOOST_REQUIRE(
        contains(bgInfo.expectedInputs,
                 ExpectedConnection{sg_d, ExpectedConnectionType::Fwd}));
    BOOST_REQUIRE(
        contains(bgInfo.expectedInputs,
                 ExpectedConnection{sg_d, ExpectedConnectionType::FwdGrad}));

    // Check a', b' only in expectedOutputs.
    BOOST_REQUIRE(bgInfo.expectedOutputs.size() == 2);
    BOOST_REQUIRE(
        contains(bgInfo.expectedOutputs,
                 ExpectedConnection{sg_a, ExpectedConnectionType::FwdGrad}));
    BOOST_REQUIRE(
        contains(bgInfo.expectedOutputs,
                 ExpectedConnection{sg_b, ExpectedConnectionType::FwdGrad}));

    /* Test ir has a bwd sg */

    auto tw_bg = tw_ir.hasGraph(bgInfo.bwdGraphId, Require::MustBeTrue);

    /* Test bg has correct inputs and outputs. */

    // The forward tensors we need must have been cloned into the backward
    // graph, regardless of whether we did recompute or fwdoutput stitching.

    auto bg_d_grad = fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_d);
    auto bg_d      = fwdIdToClonedBwdId(sg, tw_bg->unwrap(), sg_d);
    auto bg_a_grad = fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_a);
    auto bg_b_grad = fwdIdToBwdGradId(sg, tw_bg->unwrap(), sg_b);

    tw_bg->inputs().hasExactIds({bg_d_grad, bg_d}, Require::MustBeTrue);
    tw_bg->outputs().hasExactIds({bg_a_grad, bg_b_grad}, Require::MustBeTrue);

    /* Test bg TertiaryGradOp and edge grads and sums */

    // d, d' -> TertiaryGradOp
    auto tw_tertiary_grad = tw_bg->ops().hasOp<TertiaryGradOp>(
        [&](auto &tw_tertiary_grad) -> bool {
          return tw_tertiary_grad.inputs().hasIdAtIndex(
                     TertiaryGradOp::getGradInIndex(), bg_d_grad) &&
                 tw_tertiary_grad.inputs().hasIdAtIndex(
                     TertiaryGradOp::getFwdOutInIndex(), bg_d);
        },
        Require::MustBeTrue);

    // TertiaryGradOp -> edge_t; for t = a, b, c

    auto tw_edge_grad_a = tw_tertiary_grad->outputs()
                              .hasIndex(TertiaryGradOp::getArg0GradOutIndex(),
                                        Require::MustBeTrue)
                              ->tensor();

    auto tw_edge_grad_b = tw_tertiary_grad->outputs()
                              .hasIndex(TertiaryGradOp::getArg1GradOutIndex(),
                                        Require::MustBeTrue)
                              ->tensor();

    auto tw_edge_grad_c = tw_tertiary_grad->outputs()
                              .hasIndex(TertiaryGradOp::getArg2GradOutIndex(),
                                        Require::MustBeTrue)
                              ->tensor();

    // edge_t -> Sum -> t'; for t = a, b

    tw_edge_grad_a.consumers().hasOp<SumOp>(
        [&](auto &tw_sum) -> bool {
          return tw_sum.outputs().hasExactIds({bg_a_grad});
        },
        Require::MustBeTrue);

    tw_edge_grad_b.consumers().hasOp<SumOp>(
        [&](auto &tw_sum) -> bool {
          return tw_sum.outputs().hasExactIds({bg_b_grad});
        },
        Require::MustBeTrue);

    // No edge_c -> Sum -> c' exists

    tw_edge_grad_c.consumers().hasOp<SumOp>(Require::MustBeFalse);
  };

  for (auto stitchStrat : {AutodiffStitchStrategy::SafeAddFwdOutputs,
                           AutodiffStitchStrategy::AddFwdOutputs,
                           AutodiffStitchStrategy::RecomputeAllNonInputs,
                           AutodiffStitchStrategy::RecomputeMinimal}) {
    BOOST_TEST_MESSAGE("TestDoesNotPruneUnrequiredGradTensorIfProducingGradOpSt"
                       "illRequired: "
                       << stitchStrat);
    test(stitchStrat);
  }
}
