/// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_UNITTESTS_TESTUTIL_IRQUERY_TESTOP_HPP_
#define POPART_TESTS_UNITTESTS_TESTUTIL_IRQUERY_TESTOP_HPP_

#include <map>
#include <memory>
#include <utility>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
namespace irquery {

using Op                 = popart::Op;
using OperatorIdentifier = popart::OperatorIdentifier;

/**
 * A simple test op class for use in tests in this file.
 **/
class TestOp : public Op {
public:
  TestOp(const Op::Settings &settings)
      : Op(OperatorIdentifier("TestOps", "TestOp", 1), settings) {}

  void setup() override {
    for (const auto &entry : input->tensorIdMap()) {
      outInfo(entry.first) = inInfo(entry.first);
    }
  }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<TestOp>(*this);
  }

  float getSubgraphValue() const override { return getLowSubgraphValue(); }
};

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_UNITTESTS_TESTUTIL_IRQUERY_TESTOP_HPP_
