// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_OP_DUMMY_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_OP_DUMMY_HPP_

#include <limits>
#include <memory>
#include <popart/op.hpp>

#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {
class Graph;
} // namespace popart

namespace test_graphs {

namespace CustomOperators {
const popart::OperatorIdentifier Dummy_1 = {
    "dummy_domain",
    "dummy_type",
    1, // version
    popart::NumInputs{0, std::numeric_limits<int>::max()},
    1 // num outputs
};
}

class DummyOp : public popart::Op {
public:
  DummyOp(popart::Graph &graph, const popart::Op::Settings &);

  popart::InIndex getNextInIndex() { return nextInIndex++; }
  static popart::OutIndex getOutIndex() { return 0; }

  void setup() override;

  float getSubgraphValue() const override { return getLowSubgraphValue(); }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<DummyOp>(*this);
  }

private:
  popart::InIndex nextInIndex{0};
};

} // namespace test_graphs

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_TEST_GRAPHS_OP_DUMMY_HPP_
