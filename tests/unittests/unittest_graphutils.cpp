// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GraphUtilsTest

#include <testutil/test_graphs/graph_test_models.hpp>

#include <boost/test/unit_test.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/op/slice.hpp>
#include <popart/tensor.hpp>

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

  std::set<Op *> opsToOrder;
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
