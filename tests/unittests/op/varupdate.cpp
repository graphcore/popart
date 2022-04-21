// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Op_VarUpdate
#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulatorscale.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensorinfo.hpp>

using namespace popart;

namespace {

namespace TestCustomOperators {
const OperatorIdentifier TestVarUpdateOp("Test", "VarUpdateOp", 1, 1, 1);
}

// A stub VarUpdateOp used for testing.
class TestVarUpdateOp final : public VarUpdateOp {
public:
  TestVarUpdateOp(const OperatorIdentifier &opid, const Op::Settings &settings)
      : VarUpdateOp(opid, settings) {}

  std::map<InIndex, TensorId> optimizerInputs() const final { return {}; }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<TestVarUpdateOp>(*this);
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace

BOOST_AUTO_TEST_CASE(TestVarUpdateCtorSchedulePriority) {
  const auto test = [](const double expectedPriority,
                       const SessionOptions &opts) {
    Ir ir;
    ir.setUserOptions(opts);
    Graph &g = ir.getMainGraph();

    const TensorInfo tInfo{DataType::FLOAT, Shape{2, 2}};
    const std::vector<float> wData(tInfo.nelms());
    const TensorId w = "w";
    g.getTensors().addVarInit(w, tInfo, wData.data(), "w");

    const TensorId w2 = "w2";

    // Use any VarUpdateOp
    const auto op = g.createConnectedOp<TestVarUpdateOp>(
        {{AccumulatorScaleOp::getVarToUpdateInIndex(), w}},
        {{AccumulatorScaleOp::getUpdatedVarOutIndex(), w2}},
        TestCustomOperators::TestVarUpdateOp,
        Op::Settings{g, "AccumScale"});

    const auto tol = boost::test_tools::tolerance(1e-10);
    BOOST_TEST(op->settings.schedulePriority == expectedPriority, tol);
  };

  const auto optsFactory = [](const bool delayVarUpdates,
                              const int bsFactor,
                              const int phases,
                              const bool explicitIr) -> SessionOptions {
    SessionOptions opts;
    opts.delayVarUpdates                   = delayVarUpdates;
    opts.batchSerializationSettings.factor = bsFactor;
    opts.executionPhaseSettings.phases     = phases;
    opts.explicitRecomputation             = explicitIr;
    opts.enableExplicitMainLoops           = explicitIr;

    return opts;
  };

  // Implicit Ir && phases < 2 && bs factor < 2 => -inf
  test(std::numeric_limits<double>::lowest(), optsFactory(true, 1, 1, false));
  // Explicit Ir => 0.0
  test(0.0, optsFactory(true, 1, 1, true));
  // phases or bs factor >= 2 => 0.0
  test(0.0, optsFactory(true, 2, 1, false));
  test(0.0, optsFactory(true, 1, 2, false));
  // delayVarUpdates = false => 0.0
  test(0.0, optsFactory(false, 1, 1, false));
}
