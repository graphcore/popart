// Copyright(c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseAccumulatePatternTests
#include <boost/test/unit_test.hpp>

#include <testutil/irquery/irquery.hpp>

#include <popart/patterns/sparseaccumulatepattern.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/identity.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>

#include <typeindex>

using namespace popart;
using popart::irquery::Require;

BOOST_AUTO_TEST_CASE(TestPatternNamesContainsSparseAccumulatePattern) {
  BOOST_REQUIRE_NO_THROW(PatternNames::getName<SparseAccumulatePattern>());
}

BOOST_AUTO_TEST_CASE(TestPatternsEnabledDisabledApiWorks) {
  Patterns ps;

  // On by default.
  BOOST_REQUIRE(ps.isSparseAccumulateEnabled());

  // Calling disable works correctly.
  ps.enablePattern(std::type_index(typeid(SparseAccumulatePattern)), false);
  BOOST_REQUIRE(!ps.isPatternEnabled("SparseAccumulate"));

  // Calling enable (through another api) works correctly.
  ps.enablePattern("SparseAccumulate", true);
  BOOST_REQUIRE(
      ps.isPatternEnabled(std::type_index(typeid(SparseAccumulatePattern))));

  // Calling disable (through another api) works correctly.
  ps.enableSparseAccumulate(false);
  BOOST_REQUIRE(!ps.isSparseAccumulateEnabled());
}

namespace {
class TestGraph;

using AccumulateOpCreator = std::function<AccumulateBaseOp *(TestGraph &)>;

/**
 * @brief Builds a minimal graph containing a GatherGrad -> AccumulateOp. The
 * user passes in an AccumulateOpCreator that creates the AccumulateBaseOp of
 * their choice.
 *
 * The graph created is:
 *   axis = 0
 *   indices = const()
 *   w = var()
 *   yGrad = const()
 *
 *   y = gather(w, axis, indices)
 *   wGrad = gatherGrad(yGrad, axis, indices)
 *   wUpdated = accum(w, wGrad)
 */
class TestGraph {
public:
  TestGraph(const AccumulateOpCreator &mkAccOp);

  const AccumulateBaseOp *accumOp() const & { return accum; }
  GatherGradOp *gatherGradOp() & { return gatherGrad; }

  Graph &graph() & { return ir_.getMainGraph(); }
  Ir &ir() & { return ir_; }

  const int axis          = 0;
  const TensorId indices  = "indices";
  const TensorId w        = "w";
  const TensorId y        = "y";     // Gathered w
  const TensorId yGrad    = "yGrad"; // Constant 1.
  const TensorId wGrad    = "wGrad";
  const TensorId wUpdated = "wUpdated";

private:
  Ir ir_;

  // Data for constants and variables.
  const std::vector<int> indicesHost = {0};
  const std::vector<float> yGradHost = {1.f};
  const std::vector<float> wHost     = {4.f, 3.f, 2.f, 1.f};

  GatherGradOp *gatherGrad;
  AccumulateBaseOp *accum;
};

TestGraph::TestGraph(const AccumulateOpCreator &mkAccOp) {
  Graph &g = ir_.getMainGraph();

  g.getTensors().addConstInit(indices,
                              TensorInfo{DataType::INT32, Shape{1}},
                              indicesHost.data(),
                              "indices");

  g.getTensors().addVarInit(
      w, TensorInfo{DataType::FLOAT, Shape{2, 2}}, wHost.data(), "w");

  auto gather = g.createConnectedOp<GatherOp>(
      {{GatherOp::indicesInIndex(), indices}, {GatherOp::dataInIndex(), w}},
      {{GatherOp::outIndex(), y}},
      Onnx::Operators::Gather_11,
      axis,
      Op::Settings{g, "Gather"});
  gather->setup();

  g.getTensors().addConstInit(
      yGrad, TensorInfo{DataType::FLOAT, Shape{1}}, yGradHost.data(), "yGrad");

  this->gatherGrad = g.createConnectedOp<GatherGradOp>(
      {{GatherGradOp::gradInIndex(), yGrad},
       {GatherGradOp::indicesInIndex(), indices}},
      {{GatherGradOp::gradOutIndex(), wGrad}},
      *gather,
      axis);
  gatherGrad->setup();

  this->accum = mkAccOp(*this);
  accum->setup();
}

} // namespace

BOOST_AUTO_TEST_CASE(TestMatchesGatherGradFollowedByAccumulateOnly) {
  /*
    Build minimal Graph containing a Gather, GatherGrad and Accumulate, then
    test if the pattern matches on the GatherGrad.
   */
  const auto test = [](const bool expectMatches,
                       const AccumulateOpCreator &mkAccOp) {
    TestGraph tg(mkAccOp);

    // Test
    SparseAccumulatePattern pat;
    BOOST_REQUIRE_EQUAL(pat.matches(tg.gatherGradOp()), expectMatches);
  };

  // Test with valid AccumulateOp matches.
  test(true, [](TestGraph &tg) {
    return tg.graph().createConnectedOp<AccumulateOp>(
        {{AccumulateOp::getVarToUpdateInIndex(), tg.w},
         {AccumulateOp::getUpdaterInIndex(), tg.wGrad}},
        {{AccumulateOp::getUpdatedVarOutIndex(), tg.wUpdated}},
        AccumulationType::DampenedAdd,
        OptimizerValue{0.01f, true},
        Op::Settings{tg.graph(), "Accumulate"});
  });

  // Test AccumulateOp with an AccumulationType not supported by
  // SparseAccumulateOp does not match.
  test(false, [](TestGraph &tg) {
    return tg.graph().createConnectedOp<AccumulateOp>(
        {{AccumulateOp::getVarToUpdateInIndex(), tg.w},
         {AccumulateOp::getUpdaterInIndex(), tg.wGrad}},
        {{AccumulateOp::getUpdatedVarOutIndex(), tg.wUpdated}},
        AccumulationType::MovingAverageSquare,
        OptimizerValue{0.01f, true},
        Op::Settings{tg.graph(), "BadAccumulate"});
  });

  // Test a non-AccumulateOp does not match.
  test(false, [](TestGraph &tg) {
    return tg.graph().createConnectedOp<RescaleAccumulateOp>(
        {{RescaleAccumulateOp::getVarToUpdateInIndex(), tg.w},
         {RescaleAccumulateOp::getUpdaterInIndex(), tg.wGrad}},
        {{RescaleAccumulateOp::getUpdatedVarOutIndex(), tg.wUpdated}},
        AccumulationType::MovingAverageSquare,
        OptimizerValue{0.01f, true},
        Op::Settings{tg.graph(), "NotAnAccumulate"});
  });

  // Test does not match when dW has more than one consumer, even if the
  // AccumulateOp is otherwise valid. This because it is then no longer valid to
  // remove wGrad.
  test(false, [](TestGraph &tg) {
    tg.graph().createConnectedOp<IdentityOp>(
        {{IdentityOp::getInIndex(), tg.wGrad}},
        {{IdentityOp::getOutIndex(), "wGrad-id"}},
        Onnx::Operators::Identity_1,
        Op::Settings{tg.graph(), "Identity"});

    return tg.graph().createConnectedOp<AccumulateOp>(
        {{AccumulateOp::getVarToUpdateInIndex(), tg.w},
         {AccumulateOp::getUpdaterInIndex(), tg.wGrad}},
        {{AccumulateOp::getUpdatedVarOutIndex(), tg.wUpdated}},
        AccumulationType::DampenedAdd,
        OptimizerValue{0.01f, true},
        Op::Settings{tg.graph(), "Accumulate"});
  });
}

BOOST_AUTO_TEST_CASE(TestApplyCreatesCorrectIr) {
  TestGraph tg([](TestGraph &tg) {
    return tg.graph().createConnectedOp<AccumulateOp>(
        {{AccumulateOp::getVarToUpdateInIndex(), tg.w},
         {AccumulateOp::getUpdaterInIndex(), tg.wGrad}},
        {{AccumulateOp::getUpdatedVarOutIndex(), tg.wUpdated}},
        AccumulationType::DampenedAdd,
        OptimizerValue{0.01f, true},
        Op::Settings{tg.graph(), "Accumulate"});
  });

  // Test
  SparseAccumulatePattern pat;
  BOOST_REQUIRE(pat.matches(tg.gatherGradOp()));
  BOOST_REQUIRE(pat.apply(tg.gatherGradOp()));

  irquery::IrTestWrapper tw_ir(tg.ir());
  auto tw_mainGraph = tw_ir.hasGraph(tg.graph().id, Require::MustBeTrue);

  const auto i = [](const auto idx) -> int { return static_cast<int>(idx); };

  auto tw_spAccOp = tw_mainGraph->ops().hasOp<SparseAccumulateOp>(
      [&tg, &i](auto &tw_op) -> bool {
        const SparseAccumulateOp *spAccOp = tw_op.unwrap();

        return tw_op.inputs().hasIdAtIndex(
                   i(SparseAccumulateOp::getVarToUpdateInIndex()), tg.w) &&
               tw_op.inputs().hasIdAtIndex(
                   i(SparseAccumulateOp::getUpdaterInIndex()), tg.yGrad) &&
               tw_op.inputs().hasIdAtIndex(
                   i(SparseAccumulateOp::getIndicesInIndex()), tg.indices) &&
               tw_op.outputs().hasIdAtIndex(
                   i(SparseAccumulateOp::getUpdatedVarOutIndex()),
                   tg.wUpdated) &&
               (spAccOp->getAccumulationType() ==
                tg.accumOp()->getAccumulationType()) &&
               (spAccOp->getFactor() == tg.accumOp()->getFactor());
      },
      Require::MustBeTrue);

  BOOST_REQUIRE(tw_spAccOp.has_value());

  tw_mainGraph->ops().hasOp<AccumulateOp>(Require::MustBeFalse);
  tw_mainGraph->ops().hasOp<GatherGradOp>(Require::MustBeFalse);
  // TODO(T43085): Implement a native TensorsTestWrapper instead of unwrapping.
  BOOST_REQUIRE(!tw_mainGraph->unwrap().get().getTensors().contains(tg.wGrad));
}
