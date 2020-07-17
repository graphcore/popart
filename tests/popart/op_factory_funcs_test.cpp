// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NoGradOpTest

#include <boost/test/unit_test.hpp>

#include <memory>

#include <popart/op.hpp>
#include <popart/opmanager.hpp>

using namespace popart;

namespace CustomOperators {
const OperatorIdentifier DontTrain = {"com.acme", "TestOp", 1};
} // namespace CustomOperators

class TestOp : public Op {
public:
  TestOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
      : Op(_opid, settings_) {}

  void setup() final { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<TestOp>(*this);
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static popart::OpDefinition dontTrainOpDef(
    {popart::OpDefinition::Inputs({
         {"input", {{popart::DataType::FLOAT}}},
     }),
     popart::OpDefinition::Outputs({{"output", {{popart::DataType::FLOAT}}}}),
     popart::OpDefinition::Attributes({})});

// Test that the LegacyOpFactoryFunction still works.
// This test will throw an error at compile time if it doesn't.
static OpCreator<TestOp> donttrainOpCreator(
    {{CustomOperators::DontTrain, dontTrainOpDef}},
    [](const OperatorIdentifier &opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      return std::unique_ptr<TestOp>(new TestOp(opid, settings));
    });

BOOST_AUTO_TEST_CASE(Basic0) {
  logging::debug("If this test built, then LegacyOpFactoryFunc still works.");
}
