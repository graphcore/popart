// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PatternTest

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/scheduler.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>
#include <popart/topocons.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Pattern_transferBaseProperties) {
  // Check that Pattern::transferBaseProperties transfers scope, recomputeType
  // and tensorLocation.

  // Subclass from Pattern to make transferBaseProperties visible.
  class TestPattern : public Pattern {
  public:
    TestPattern()          = default;
    virtual ~TestPattern() = default;

    void testTransferBaseProperties(Op *from, Op *to) const {
      Pattern::transferBaseProperties(from, to);
    }
  };

  // Subclass from Op to be able to instantiate Ops.
  class TestOp : public Op {
  public:
    TestOp(const OperatorIdentifier &_opid, const Op::Settings &settings)
        : Op(_opid, settings) {}

    // These functions make Op abstract, but we don't need them for this test.
    virtual std::unique_ptr<Op> clone() const {
      return std::make_unique<TestOp>(this->opid, this->settings);
    };
    virtual float getSubgraphValue() const { return 0.0f; };
  };

  Ir ir{};
  GraphId graphId{"test_graph"};

  OperatorIdentifier opId{"test_ops", "test_op", 1};

  Graph graph{ir, graphId};

  Op::Settings op1Settings{graph, "test_op1"};
  op1Settings.scope          = Scope{} / "scope_1";
  op1Settings.recomputeType  = RecomputeType::Checkpoint;
  op1Settings.tensorLocation = TensorStorage::OffChip;

  Op::Settings op2Settings{graph, "test_op2"};
  op2Settings.scope          = Scope{} / "scope_2";
  op2Settings.recomputeType  = RecomputeType::Recompute;
  op2Settings.tensorLocation = TensorStorage::OnChip;

  TestOp op1{opId, op1Settings};
  TestOp op2{opId, op2Settings};

  TestPattern pattern{};

  BOOST_CHECK(op1.settings.scope != op2.settings.scope);
  BOOST_CHECK(op1.settings.recomputeType != op2.settings.recomputeType);
  BOOST_CHECK(op1.settings.tensorLocation != op2.settings.tensorLocation);

  pattern.testTransferBaseProperties(&op1, &op2);

  BOOST_CHECK(op1.settings.scope == op2.settings.scope);
  BOOST_CHECK(op1.settings.recomputeType == op2.settings.recomputeType);
  BOOST_CHECK(op1.settings.tensorLocation == op2.settings.tensorLocation);
}
