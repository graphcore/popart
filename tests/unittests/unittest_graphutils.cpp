// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/logging.hpp"
#include "popart/pointercomparators.hpp"
#include "popart/util.hpp"

#define BOOST_TEST_MODULE GraphUtilsTest

#include <boost/test/unit_test.hpp>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <testutil/test_graphs/graph_test_models.hpp>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/slice.hpp>
#include <popart/tensor.hpp>

#include "popart/tensordebuginfo.hpp"

using namespace popart;
using namespace graphutils;

namespace {

void test_traversal(TraversalType traversalType,
                    TraversalDirection traversalDirection,
                    std::vector<Tensor *> start,
                    std::vector<TensorId> expected) {
  auto it = expected.begin();

  traverse(
      start,
      [&it, &expected](Tensor *t) {
        logging::trace("Visiting: {}", t->id);
        if (!expected.empty()) {
          if (it != expected.end()) {
            BOOST_CHECK_EQUAL(t->id, *it++);
          } else {
            BOOST_CHECK(false);
          }
        }
        return true;
      },
      [](Op *op, Tensor *t0, Tensor *t1) { return true; },
      traversalType,
      VisitType::Pre,
      traversalDirection);

  // Check if end of expected was reached
  BOOST_CHECK_EQUAL(std::distance(expected.begin(), it),
                    std::distance(expected.begin(), expected.end()));
}

} // namespace

BOOST_AUTO_TEST_CASE(TraverseBFSFwd) {
  GraphTestModel1 model;

  std::vector<Tensor *> start{model.getIr().getTensor("t0")};

  std::vector<TensorId> expected{"t0",
                                 "t1",
                                 "t2",
                                 "t3",
                                 "sub0/t1",
                                 "t6",
                                 "sub1/t2",
                                 "sub0/t3",
                                 "sub0/t7",
                                 "t5",
                                 "sub1/t8",
                                 "t7",
                                 "t8"};

  test_traversal(TraversalType::BreadthFirst,
                 TraversalDirection::Forward,
                 start,
                 expected);
}

BOOST_AUTO_TEST_CASE(TraverseBFSBwd) {
  GraphTestModel1 model;

  std::vector<Tensor *> start{model.getIr().getTensor("t5"),
                              model.getIr().getTensor("t7"),
                              model.getIr().getTensor("t8")};

  std::vector<TensorId> expected{"t5",
                                 "t7",
                                 "t8",
                                 "t4",
                                 "t6",
                                 "sub0/t7",
                                 "sub1/t8",
                                 "t1",
                                 "t3",
                                 "sub0/t3",
                                 "sub0/t1",
                                 "sub1/t2",
                                 "t0",
                                 "t2"};

  test_traversal(TraversalType::BreadthFirst,
                 TraversalDirection::Backward,
                 start,
                 expected);
}

BOOST_AUTO_TEST_CASE(TraverseDFSFwd) {
  GraphTestModel1 model;

  std::vector<Tensor *> start{model.getIr().getTensor("t0")};

  std::vector<TensorId> expected{"t0",
                                 "t3",
                                 "t6",
                                 "t5",
                                 "sub0/t3",
                                 "sub0/t7",
                                 "t7",
                                 "t2",
                                 "sub1/t2",
                                 "sub1/t8",
                                 "t8",
                                 "t1",
                                 "sub0/t1"};

  test_traversal(
      TraversalType::DepthFirst, TraversalDirection::Forward, start, expected);
}

BOOST_AUTO_TEST_CASE(TraverseDFSBwd) {
  GraphTestModel1 model;

  std::vector<Tensor *> start{model.getIr().getTensor("t5"),
                              model.getIr().getTensor("t7"),
                              model.getIr().getTensor("t8")};

  std::vector<TensorId> expected{"t8",
                                 "sub1/t8",
                                 "sub1/t2",
                                 "t2",
                                 "t0",
                                 "t7",
                                 "sub0/t7",
                                 "sub0/t1",
                                 "t1",
                                 "sub0/t3",
                                 "t3",
                                 "t5",
                                 "t6",
                                 "t4"};

  test_traversal(
      TraversalType::DepthFirst, TraversalDirection::Backward, start, expected);
}

BOOST_AUTO_TEST_CASE(TraverseBFSFwdBwd) {
  GraphTestModel1 model;

  std::vector<Tensor *> start{model.getIr().getTensor("sub1/t8")};

  std::vector<TensorId> expected{"sub1/t8",
                                 "t8",
                                 "sub1/t2",
                                 "t2",
                                 "t0",
                                 "t1",
                                 "t3",
                                 "sub0/t1",
                                 "t6",
                                 "sub0/t3",
                                 "sub0/t7",
                                 "t5",
                                 "t7",
                                 "t4"};

  test_traversal(TraversalType::BreadthFirst,
                 TraversalDirection::ForwardBackward,
                 start,
                 expected);
}

BOOST_AUTO_TEST_CASE(TraverseBFSBwdFwd) {
  GraphTestModel1 model;

  std::vector<Tensor *> start{model.getIr().getTensor("sub1/t8")};

  std::vector<TensorId> expected{"sub1/t8",
                                 "sub1/t2",
                                 "t8",
                                 "t2",
                                 "t0",
                                 "t1",
                                 "t3",
                                 "sub0/t1",
                                 "t6",
                                 "sub0/t3",
                                 "sub0/t7",
                                 "t5",
                                 "t7",
                                 "t4"};

  test_traversal(TraversalType::BreadthFirst,
                 TraversalDirection::BackwardForward,
                 start,
                 expected);
}

BOOST_AUTO_TEST_CASE(TraverseDFSFwdBwd) {
  GraphTestModel1 model;

  std::vector<Tensor *> start{model.getIr().getTensor("sub1/t8")};

  std::vector<TensorId> expected{"sub1/t8",
                                 "t8",
                                 "sub1/t2",
                                 "t2",
                                 "t0",
                                 "t3",
                                 "t6",
                                 "t5",
                                 "t4",
                                 "sub0/t3",
                                 "sub0/t7",
                                 "t7",
                                 "sub0/t1",
                                 "t1"};

  test_traversal(TraversalType::DepthFirst,
                 TraversalDirection::ForwardBackward,
                 start,
                 expected);
}

BOOST_AUTO_TEST_CASE(TraverseDFSBwdFwd) {
  GraphTestModel1 model;

  std::vector<Tensor *> start{model.getIr().getTensor("sub1/t8")};

  std::vector<TensorId> expected{"sub1/t8",
                                 "sub1/t2",
                                 "t2",
                                 "t0",
                                 "t3",
                                 "t6",
                                 "t5",
                                 "t4",
                                 "sub0/t3",
                                 "sub0/t7",
                                 "sub0/t1",
                                 "t7",
                                 "t1",
                                 "t8"};

  test_traversal(TraversalType::DepthFirst,
                 TraversalDirection::BackwardForward,
                 start,
                 expected);
}

BOOST_AUTO_TEST_CASE(TestOpsWithBefores) {
  GraphTestModel1 model;
  auto &ops = model.getIr().getMainGraph().getOps();

  std::set<Op *, POpCmp> opsToOrder;
  for (auto &op : ops) {
    opsToOrder.insert(op.second.get());
  }

  auto opsWithBefores = getOpsWithBefores(opsToOrder);

  std::map<std::string, std::set<std::string>> expected;
  expected.insert({"Concat", {"Slice0", "Slice1", "Slice2"}});
  expected.insert({"Slice0", {"Slice1", "Slice2"}});
  expected.insert({"Call0", {"Slice0", "Slice1", "Slice2"}});
  expected.insert({"Call1", {"Slice1"}});
  expected.insert({"AddLhsInplace", {"Concat", "Slice0", "Slice1", "Slice2"}});
  expected.insert({"Slice2", {}});
  expected.insert({"Slice1", {}});

  for (auto &op : opsWithBefores) {

    auto it0 = expected.find(op.first->name());

    BOOST_CHECK(it0 != expected.end());

    BOOST_CHECK_EQUAL(it0->second.size(), op.second.size());

    std::set<std::string> beforeNames;
    for (Op *before : op.second) {
      beforeNames.insert(before->name());
      auto it1 = it0->second.find(before->name());
      BOOST_CHECK(it1 != it0->second.end());
    }
    logging::trace("Op: {} befores: {}", op.first->debugName(), beforeNames);
  }
}

BOOST_AUTO_TEST_CASE(TestFindMatchingOps) {
  GraphTestModel1 model;
  auto &graph = model.getIr().getMainGraph();

  model.getIr().dotCheckpoint(model.getIr(), "Final");

  {
    graphutils::OpPreds preds{
        [](const Op *op) { return op->isConvertibleTo<SliceOp>(); },
        [](const Op *op) { return op->isConvertibleTo<CallOp>(); }};
    graphutils::Edges edges{
        {0, 1},
    };

    auto matches = graphutils::findMatchingOps(graph, preds, edges);
    BOOST_REQUIRE_EQUAL(matches.size(), 2);
    BOOST_REQUIRE_EQUAL(matches.at(0).size(), 2);
    BOOST_REQUIRE_EQUAL(matches.at(1).size(), 2);

    BOOST_REQUIRE(matches.at(0).at(0)->isConvertibleTo<SliceOp>());
    BOOST_REQUIRE(matches.at(1).at(0)->isConvertibleTo<SliceOp>());
    BOOST_REQUIRE(matches.at(0).at(1)->isConvertibleTo<CallOp>());
    BOOST_REQUIRE(matches.at(1).at(1)->isConvertibleTo<CallOp>());

    BOOST_REQUIRE_NE(matches.at(0).at(0), matches.at(1).at(0));
    BOOST_REQUIRE_NE(matches.at(0).at(1), matches.at(1).at(1));
  }

  {
    graphutils::OpPreds preds{
        [](const Op *op) { return op->isConvertibleTo<SliceInplaceOp>(); },
        [](const Op *op) { return op->isConvertibleTo<CallOp>(); }};
    graphutils::Edges edges{
        {0, 1},
    };

    auto matches = graphutils::findMatchingOps(graph, preds, edges);

    BOOST_REQUIRE_EQUAL(matches.size(), 1);
    BOOST_REQUIRE_EQUAL(matches.at(0).size(), 2);

    BOOST_REQUIRE(matches.at(0).at(0)->isConvertibleTo<SliceInplaceOp>());
    BOOST_REQUIRE(matches.at(0).at(1)->isConvertibleTo<CallOp>());
  }

  {
    graphutils::OpPreds preds{
        [](const Op *op) {
          return op->isConvertibleTo<SliceOp>() ||
                 op->isConvertibleTo<SliceInplaceOp>();
        },
        [](const Op *op) { return op->isConvertibleTo<CallOp>(); }};
    graphutils::Edges edges{
        {0, 1, 0, -1},
    };

    auto matches = graphutils::findMatchingOps(graph, preds, edges);
    BOOST_REQUIRE_EQUAL(matches.size(), 3);
    BOOST_REQUIRE_EQUAL(matches.at(0).size(), 2);
    BOOST_REQUIRE_EQUAL(matches.at(1).size(), 2);
    BOOST_REQUIRE_EQUAL(matches.at(2).size(), 2);
  }

  {
    graphutils::OpPreds preds{
        [](const Op *op) { return true; },
        [](const Op *op) { return true; },
    };
    graphutils::Edges edges{
        {0, 1, 0, 0},
        {1, 0, 0, 0},
    };

    auto matches = graphutils::findMatchingOps(graph, preds, edges);
    BOOST_REQUIRE_EQUAL(matches.size(), 0);
  }
}

BOOST_AUTO_TEST_CASE(TestTraverseCallSites) {
  TraverseCallSiteTestModel model;
  auto &graph                 = model.getIr().getMainGraph();
  std::vector<Tensor *> start = {graph.getTensor("t0")};

  {
    bool visited_t2 = false;
    auto visit      = [&visited_t2](Tensor *t) {
      if (t->id == "t2") {
        visited_t2 = true;
      }
      return true;
    };
    graphutils::traverse(
        start,
        visit,
        [](auto, auto, auto) { return true; },
        graphutils::TraversalType::DepthFirst,
        graphutils::VisitType::Pre,
        graphutils::TraversalDirection::ForwardBackward,
        graphutils::TraverseCallSites::All);
    BOOST_REQUIRE(visited_t2);
  }
  {
    bool visited_t2 = false;
    auto visit      = [&visited_t2](Tensor *t) {
      if (t->id == "t2") {
        visited_t2 = true;
      }
      return true;
    };
    graphutils::traverse(
        start,
        visit,
        [](auto, auto, auto) { return true; },
        graphutils::TraversalType::DepthFirst,
        graphutils::VisitType::Pre,
        graphutils::TraversalDirection::ForwardBackward,
        graphutils::TraverseCallSites::Current);
    BOOST_REQUIRE(!visited_t2);
  }
}
