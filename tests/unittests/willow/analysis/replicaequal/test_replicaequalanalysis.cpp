// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_ReplicaEqualAnalysis
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <map>
#include <string>
#include <testutil/test_graphs/graph_test_models.hpp>
#include <vector>
#include <popart/analysis/replicaequal/replicaequalanalysis.hpp>
#include <popart/graph.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/subtract.hpp>
#include <popart/sessionoptions.hpp>

#include "popart/graphid.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/replicatedstreammode.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp" // IWYU pragma: keep
#include "popart/util.hpp"
#include "popart/vendored/any.hpp" // IWYU pragma: keep
#include "popart/vendored/optional.hpp"
#include "testutil/irquery/irquery.hpp"

using namespace popart;
using namespace popart::irquery;

namespace {

/**
 * Get the value of the tensor for all consumers/producers. Note that in general
 * these values may not agree but it's convenient to get the value this way
 * for simple models. Note we fail the test if values don't agree.
 **/
void simpleRequireValue(const Ir &ir,
                        const ReplicaEqualAnalysis &analysis,
                        const TensorId &t,
                        IsReplicaEqual expValue) {
  auto tensor = ir.getTensor(t);

  // Check replica-equal values for all producers and consumers.
  if (tensor->hasProducer()) {
    Op *producer = tensor->getProducer();
    auto actValue =
        analysis.isOpOutputEqual(producer, producer->outIndex(tensor));

    BOOST_REQUIRE_MESSAGE(
        expValue == actValue,
        logging::format("Expected {} to be {} (as produced by '{}')",
                        t,
                        expValue,
                        producer->str()));
  }
  for (auto &consumer : tensor->consumers.getOps()) {
    auto actValue =
        analysis.isOpInputEqual(consumer, consumer->inIndex(tensor));

    BOOST_REQUIRE_MESSAGE(
        expValue == actValue,
        logging::format("Expected {} to be {} (as consumed by '{}')",
                        t,
                        expValue,
                        consumer->str()));
  }
}
} // namespace

BOOST_AUTO_TEST_CASE(ReplicaEqualsAnalysis_hostLoad) {

  /**
   * Note, typically x would be replicated in this kind of IR, but we check the
   * analysis for both modes.
   *
   *                             | replica equal when | replica equal when
   *                             | x.mode=Broadcast?  | x.mode=Replicate?
   *           Init              |                    |
   *            |                |                    |
   *          [x__t0]  . . . . . | True               | True
   *            |                |                    |
   *            v                |                    |
   *        HostLoad             |                    |
   *  w . . . . (  . . . . . . . | True               | False
   *  |    [x__t0__t1] . . . . . | True               | False
   *  |         |                |                    |
   *  '-----.   |                |                    |
   *        v   v                |                    |
   *      Accumulate             |                    |
   *          |      [c=5.0f]  . | True               | True
   *       [w__t2] . . ( . . . . | True               | False
   *          | .------'         |                    |
   *          | |                |                    |
   *          v v                |                    |
   *          Add                |                    |
   *           |                 |                    |
   *          [y]  . . . . . . . | True               | False
   *           |                 |                    |
   *           v                 |                    |
   *       HostStore             |                    |
   **/

  {
    GraphTestModel4 model(ReplicatedStreamMode::Broadcast);
    auto &ir = model.getIr();

    // TODO (T48752): Remove.
    SessionOptions opts;
    opts._enableRngStateManagement = true;
    ir.setUserOptions(opts);

    ReplicaEqualAnalysis analysis{ir};
    analysis.apply();

    simpleRequireValue(ir, analysis, "x", true);
    simpleRequireValue(ir, analysis, "x__t0", true);
    simpleRequireValue(ir, analysis, "x__t0__t1", true);
    simpleRequireValue(ir, analysis, "w", true);
    simpleRequireValue(ir, analysis, "w__t2", true);
    simpleRequireValue(ir, analysis, "c", true);
    simpleRequireValue(ir, analysis, "y", true);
  }
  {
    GraphTestModel4 model(ReplicatedStreamMode::Replicate);
    auto &ir = model.getIr();

    // TODO (T48752): Remove.
    SessionOptions opts;
    opts._enableRngStateManagement = true;
    ir.setUserOptions(opts);

    ReplicaEqualAnalysis analysis{ir};
    analysis.apply();

    simpleRequireValue(ir, analysis, "x", false);
    simpleRequireValue(ir, analysis, "x__t0", true);
    simpleRequireValue(ir, analysis, "x__t0__t1", false);
    simpleRequireValue(ir, analysis, "w", false);
    simpleRequireValue(ir, analysis, "w__t2", false);
    simpleRequireValue(ir, analysis, "c", true);
    simpleRequireValue(ir, analysis, "y", false);
  }
}

BOOST_AUTO_TEST_CASE(ReplicaEqualsAnalysis_simpleTrain) {

  /**
   * Forwards pass:
   *
   *   [inputs]        [labels]         [ weights ]        Expect replica equal?
   *     |                 |             |       |
   *     o . . . . . . . . ( . . . . . . ( . . . ( . . . . False  = inputs
   *     |                 o . . . . . . ( . . . ( . . . . False  = labels
   *     |                 |             o . . . o . . . . True   = weights
   *     v                 |             |       |
   *    Identity           |             |       |
   *     |                 |             |       |
   *     o . . . . . . . . ( . . . . . . ( . . . ( . . . . False  = t4
   *     |    .------------(-------------'       |
   *     |----(------------(-----------.         |
   *     |    |            |           |         |
   *     v    v            |           |         |
   *     MatMul            |           |         |
   *     |                 |           |         |
   *     o . . . . . . . . ( . . . . . ( . . . . ( . . . . False  = t5
   *     | .---------------'           |         |
   *     | |                           |         |
   *     v v                           |         |
   *     Sub                           |         |
   *     |                             |         |
   *     0 . . . . . . . . . . . . . . ( . . . . ( . . . . False  = t6
   *     |---------------------.       |         |
   *     v                     |       |         |
   *     L1                    |       |         |
   *     |                     |       |         |
   *     o . . . . . . . . . . ( . . . ( . . . . ( . . . . False  = loss
   *     |                     |       |         |
   *     v                     |       |         |
   *   [loss]                  |       |         |
   *                           |       |         |
   * Backwards pass:           |       |         |
   *                           |       |         |
   *    [1]                    |       |         |
   *     |                     |       |         |
   *     o . . . . . . . . . . ( . . . ( . . . . ( . . . . True   = t7
   *     |  .------------------'       |         |
   *     v  v                          |         |
   *   L1Grad            .-------------'         |
   *     |               |                       |
   *     o . . . . . . . ( . . . . . . . . . . . ( . . . . False  = t8
   *     |               v                       |
   *     | TransposeInplace                      |
   *     |  |                                    |
   *     v  v                                    |
   *    MatMul                                   |
   *     |                                       |
   *     o . . . . . . . . . . . . . . . . . . . ( . . . . False  = t9
   *     |                                       |
   * ....(.................................      |
   * :   v                                :      |
   * :  ReplicatedAllReduce               :      |
   * :   |                                :      |
   * :   |         Outlined iff SG1::Yes  :      |
   * :...(................................:      |
   *     |                                       |
   *     o . . . . . . . . . . . . . . . . . . . ( . . . . True   = t10
   *     |         .-----------------------------'
   * ....(.........(.......................
   * :   v         v                      :
   * :  SGD0VarUpdate                     :
   * :   |                                :
   * :   |                                :
   * :   |         Outlined iff SG2::Yes  :
   * :...(................................:
   *     |
   *     o . . . . . . . . . . . . . . . . . . . . . . . . True   = t11
   *     |
   *   [...]
   *
   **/
  auto sg1s = std::vector<GraphTestModel5::SG1>(
      {GraphTestModel5::SG1::No, GraphTestModel5::SG1::Yes});
  auto sg2s = std::vector<GraphTestModel5::SG2>(
      {GraphTestModel5::SG2::No, GraphTestModel5::SG2::Yes});

  for (auto sg1 : sg1s) {
    for (auto sg2 : sg2s) {
      std::cout << "Testing sg1=" << static_cast<int>(sg1) << ", "
                << "sg2=" << static_cast<int>(sg2) << std::endl;
      GraphTestModel5 model(sg1, sg2);
      auto &ir = model.getIr();

      // TODO (T48752): Remove.
      SessionOptions opts;
      opts._enableRngStateManagement = true;
      ir.setUserOptions(opts);

      ReplicaEqualAnalysis analysis{ir};
      analysis.apply();

      // Get a test wrapper for the main graph.
      IrTestWrapper tw_ir{ir};
      auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id);

      // Find TensorIds.
      auto weights = "weights";
      auto inputs  = tw_mainGraph->ops()
                        .hasOp<IdentityOp>()
                        ->inputs()
                        .hasIndex(IdentityOp::getInIndex())
                        ->id();
      auto labels = tw_mainGraph->ops()
                        .hasOp<SubtractOp>()
                        ->inputs()
                        .hasIndex(SubtractOp::getArg1InIndex())
                        ->id();
      auto t4 = tw_mainGraph->ops()
                    .hasOp<IdentityOp>()
                    ->outputs()
                    .hasIndex(IdentityOp::getOutIndex())
                    ->id();
      auto t5 = tw_mainGraph->ops()
                    .hasOp<SubtractOp>()
                    ->inputs()
                    .hasIndex(SubtractOp::getArg0InIndex())
                    ->id();
      auto t6 = tw_mainGraph->ops()
                    .hasOp<SubtractOp>()
                    ->outputs()
                    .hasIndex(SubtractOp::getOutIndex())
                    ->id();
      auto loss = tw_mainGraph->ops()
                      .hasOp<L1Op>()
                      ->outputs()
                      .hasIndex(L1Op::getOutIndex())
                      ->id();
      auto t7 = tw_mainGraph->ops()
                    .hasOp<L1GradOp>()
                    ->inputs()
                    .hasIndex(L1GradOp::getGradInIndex())
                    ->id();
      auto t8 = tw_mainGraph->ops()
                    .hasOp<L1GradOp>()
                    ->outputs()
                    .hasIndex(L1GradOp::getOutIndex())
                    ->id();

      auto getT9 = [](GraphTestWrapper &tw_graph) {
        return tw_graph.ops()
            .hasOp<ReplicatedAllReduceOp>()
            ->inputs()
            .hasIndex(ReplicatedAllReduceOp::getInIndex())
            ->id();
      };
      auto getT10 = [](GraphTestWrapper &tw_graph) {
        return tw_graph.ops()
            .hasOp<ReplicatedAllReduceOp>()
            ->outputs()
            .hasIndex(ReplicatedAllReduceOp::getOutIndex())
            ->id();
      };

      TensorId t9;
      TensorId t10;

      if (sg1 == GraphTestModel5::SG1::No) {
        // Get t9/t10 from main graph.
        t9  = getT9(*tw_mainGraph);
        t10 = getT10(*tw_mainGraph);
      } else {
        // Get t9/t10 from sub graph.
        auto tw_sg1 = tw_ir.hasGraph(GraphId("sg1"));
        t9          = getT9(*tw_sg1);
        t10         = getT10(*tw_sg1);
      }

      auto getT11 = [](GraphTestWrapper &tw_graph) {
        return tw_graph.ops()
            .hasOp<SGD0VarUpdateOp>(Require::MustBeTrue)
            ->outputs()
            .hasIndex(SGD0VarUpdateOp::getUpdatedVarOutIndex(),
                      Require::MustBeTrue)
            ->id();
      };

      TensorId t11;

      if (sg2 == GraphTestModel5::SG2::No) {
        // Get t11 from main graph.
        t11 = getT11(*tw_mainGraph);
      } else {
        // Get t11 from sub graph.
        auto tw_sg2 = tw_ir.hasGraph(GraphId("sg2"));
        t11         = getT11(*tw_sg2);
      }

      // Check analysis results.
      simpleRequireValue(ir, analysis, inputs, false);
      simpleRequireValue(ir, analysis, labels, false);
      simpleRequireValue(ir, analysis, weights, true);
      simpleRequireValue(ir, analysis, t4, false);
      simpleRequireValue(ir, analysis, t5, false);
      simpleRequireValue(ir, analysis, t6, false);
      simpleRequireValue(ir, analysis, loss, false);
      simpleRequireValue(ir, analysis, t7, true);
      simpleRequireValue(ir, analysis, t8, false);
      simpleRequireValue(ir, analysis, t9, false);
      simpleRequireValue(ir, analysis, t10, true);
      simpleRequireValue(ir, analysis, t11, true);
    }
  }
}

BOOST_AUTO_TEST_CASE(ReplicaEqualsAnalysis_checkOpAttrs) {
  GraphTestModel5 model(GraphTestModel5::SG1::No, GraphTestModel5::SG2::No);
  auto &ir = model.getIr();

  // TODO (T48752): Remove.
  SessionOptions opts;
  opts._enableRngStateManagement = true;
  ir.setUserOptions(opts);

  ReplicaEqualAnalysis analysis{ir};
  analysis.apply();

  // Get a test wrapper for the main graph.
  IrTestWrapper tw_ir{ir};
  auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id);

  // Check getOpAttrs for Ops. Check result of operator<< because theres no
  // implementation for operator== on popart::any.
  auto getOpAttrStr = [&](Op *op) {
    std::stringstream ss;
    ss << (analysis.getOpAttrs(op));
    return ss.str();
  };

  // Check the all reduce.
  auto allReduceOp =
      tw_mainGraph->ops().hasOp<ReplicatedAllReduceOp>()->unwrap();
  BOOST_REQUIRE_EQUAL("[replEqIn0: 0, replEqOut0: 1, ]",
                      getOpAttrStr(allReduceOp));

  // Check the identity.
  auto identityOp = tw_mainGraph->ops().hasOp<IdentityOp>()->unwrap();
  BOOST_REQUIRE_EQUAL("[replEqIn0: 0, replEqOut0: 0, ]",
                      getOpAttrStr(identityOp));

  // Check the var update.
  auto varUpdateOp = tw_mainGraph->ops().hasOp<SGD0VarUpdateOp>()->unwrap();
  BOOST_REQUIRE_EQUAL("[replEqIn0: 1, replEqIn1: 1, replEqOut0: 1, ]",
                      getOpAttrStr(varUpdateOp));
}
