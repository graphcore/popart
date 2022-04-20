// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE explicitrecompute_unittest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <map>
#include <string>
#include <testutil/test_graphs/graph_test_models.hpp>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/transforms/explicitrecompute.hpp>
#include <popart/transforms/prune.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/util.hpp"

using namespace popart;
using namespace graphutils;

/**
 * Test if explicit recomputation clones the right operations and if the
 * operations in the test graph get assigned the right OpFinalLossRelation.
 */
BOOST_AUTO_TEST_CASE(TestExplicitRecomputeAndFinalLossRelation) {
  // Without pipelining: No multiple recompute
  // With pipelining: Multiple recompute
  for (bool pipelining : std::vector<bool>{false, true}) {
    ExplicitRecomputeTestModel model(pipelining, 4, 4);
    auto &graph = model.getIr().getMainGraph();

    // Check that the number of operations and their relation to the loss
    // is as expected after autodiff and before recompute
    {
      std::map<std::string, int> opNameCount;
      std::map<OpFinalLossRelation, int> relationCount;
      auto relations = graphutils::getOpFinalLossRelations(graph);

      for (auto relation : relations) {
        logging::info("Relations: {} is {}",
                      relation.first->debugName(),
                      relation.second);
        relationCount[relation.second]++;
        opNameCount[relation.first->name()]++;
      }
      logging::info("Num relations: {}", relationCount);
      logging::info("Num ops with name: {}", opNameCount);

      // 25 operations are expected to lead to the loss
      BOOST_REQUIRE_EQUAL(relationCount[OpFinalLossRelation::ToLoss], 25);

      // 107 operations are expected to lead from the loss
      BOOST_REQUIRE_EQUAL(relationCount[OpFinalLossRelation::FromLoss], 107);

      // Due to anchoring the IdentityOp output post MatMulOp, we have generated
      // 4x4 (16) FromToLoss (see ExplicitRecomputeTestModel)
      BOOST_REQUIRE_EQUAL(relationCount[OpFinalLossRelation::FromToLoss], 16);

      // Not expecting any ToFromLoss before recomputation, since all gradient
      // operations have a direct path from the loss, post-autodiff
      BOOST_REQUIRE_EQUAL(relationCount[OpFinalLossRelation::ToFromLoss], 0);

      // Count the forward operation + lhs and rhs backward operation (3),
      // which will all have the same name
      std::map<std::string, int> expected{
          {"MatMul_0_0", 3}, {"MatMul_0_1", 3}, {"MatMul_0_2", 3},
          {"MatMul_0_3", 3}, {"MatMul_1_0", 3}, {"MatMul_1_1", 3},
          {"MatMul_1_2", 3}, {"MatMul_1_3", 3}, {"MatMul_2_0", 3},
          {"MatMul_2_1", 3}, {"MatMul_2_2", 3}, {"MatMul_2_3", 3},
          {"MatMul_3_0", 3}, {"MatMul_3_1", 3}, {"MatMul_3_2", 3},
          {"MatMul_3_3", 3}, {"Add_2_0", 3},    {"Add_2_1", 3},
          {"Add_2_2", 3},    {"Add_2_3", 3},    {"Add_3_0", 3},
          {"Add_3_1", 3},    {"Add_3_2", 3},    {"Add_3_3", 3}};

      for (auto expect : expected) {
        std::tuple<TensorId, int> left{expect.first, opNameCount[expect.first]};
        std::tuple<TensorId, int> right{expect.first, expect.second};
        BOOST_REQUIRE_EQUAL(left, right);
      }
    }

    // Change the model by applying explicit recomputation
    model.getIr().applyTransform(ExplicitRecompute::id(), graph);

    // Remove superfluous operations to reduce clutter
    model.getIr().applyTransform(Prune::id(), graph);

    // Check that the number of operations and their relation to the loss
    // is as expected after recompute and pruning
    {
      auto numRecomputed = 0;
      std::map<std::string, int> opNameCount;
      std::map<OpFinalLossRelation, int> relationCount;
      auto relations = graphutils::getOpFinalLossRelations(graph);

      for (auto relation : relations) {
        logging::info("Relations: {} is {}",
                      relation.first->debugName(),
                      relation.second);
        relationCount[relation.second]++;
        opNameCount[relation.first->name()]++;

        // Ensure recomputed ops are always ToFromLoss
        if (relation.first->settings.recomputeType ==
            RecomputeType::Recomputed) {
          numRecomputed++;
          BOOST_REQUIRE_EQUAL(relation.second, OpFinalLossRelation::ToFromLoss);
        }
      }
      logging::info("Num relations: {}", relationCount);
      logging::info("Num ops with name: {}", opNameCount);

      // Expecting 20 recomputed operations
      BOOST_REQUIRE_EQUAL(numRecomputed, pipelining ? 21 : 20);

      BOOST_REQUIRE_EQUAL(relationCount[OpFinalLossRelation::ToLoss], 23);
      BOOST_REQUIRE_EQUAL(relationCount[OpFinalLossRelation::FromLoss], 107);
      // Due to anchoring the IdentityOp output post MatMulOp
      BOOST_REQUIRE_EQUAL(relationCount[OpFinalLossRelation::FromToLoss], 16);
      // Now expecting 20 (pipelining: 21) ToFromLoss after recomputation
      BOOST_REQUIRE_EQUAL(relationCount[OpFinalLossRelation::ToFromLoss],
                          pipelining ? 21 : 20);

      // Count the forward operation + lhs and rhs backward operation,
      // plus the (one or two) recomputed operations (4 or 5),
      // which will all have the same name. The first matmul of each
      // layer is not recomputed (checkpoints).
      std::map<std::string, int> expected;

      std::map<std::string, int> expected_normal{
          {"MatMul_0_0", 3}, {"MatMul_0_1", 4}, {"MatMul_0_2", 4},
          {"MatMul_0_3", 4}, {"MatMul_1_0", 3}, {"MatMul_1_1", 4},
          {"MatMul_1_2", 4}, {"MatMul_1_3", 4}, {"MatMul_2_0", 3},
          {"MatMul_2_1", 4}, {"MatMul_2_2", 4}, {"MatMul_2_3", 4},
          {"MatMul_3_0", 3}, {"MatMul_3_1", 4}, {"MatMul_3_2", 4},
          {"MatMul_3_3", 4}, {"Add_2_0", 4},    {"Add_2_1", 4},
          {"Add_2_2", 4},    {"Add_2_3", 4},    {"Add_3_0", 4},
          {"Add_3_1", 4},    {"Add_3_2", 4},    {"Add_3_3", 3}};

      // With pipelining, the outputs consumed by more than one downstream
      // pipeline stage are recomputed twice (5 total operations by the same
      // name), while the end of a layer (end of a pipeline stage) may not be
      // recomputed at all.
      std::map<std::string, int> expected_pipeline{
          {"MatMul_0_0", 3}, {"MatMul_0_1", 5}, {"MatMul_0_2", 5},
          {"MatMul_0_3", 3}, {"MatMul_1_0", 3}, {"MatMul_1_1", 5},
          {"MatMul_1_2", 5}, {"MatMul_1_3", 4}, {"MatMul_2_0", 3},
          {"MatMul_2_1", 4}, {"MatMul_2_2", 4}, {"MatMul_2_3", 3},
          {"MatMul_3_0", 3}, {"MatMul_3_1", 4}, {"MatMul_3_2", 4},
          {"MatMul_3_3", 4}, {"Add_2_0", 4},    {"Add_2_1", 4},
          {"Add_2_2", 4},    {"Add_2_3", 3},    {"Add_3_0", 4},
          {"Add_3_1", 4},    {"Add_3_2", 4},    {"Add_3_3", 3}};

      expected = pipelining ? expected_pipeline : expected_normal;

      for (auto expect : expected) {
        std::tuple<TensorId, int> left{expect.first, opNameCount[expect.first]};
        std::tuple<TensorId, int> right{expect.first, expect.second};
        BOOST_REQUIRE_EQUAL(left, right);
      }
    }
  }
}
