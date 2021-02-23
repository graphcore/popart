// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DropoutBasicTest0

#include <boost/test/unit_test.hpp>
#include <popart/ir.hpp>
#include <popart/op/dropout.hpp>
#include <popart/opmanager.hpp>

// TODO(T25465): this test may not be needed once dropout is outlineable
BOOST_AUTO_TEST_CASE(DropoutSetSeedModifierTest) {
  using namespace popart;
  Ir ir;

  std::unique_ptr<Op> op =
      OpManager::createOp(Onnx::Operators::Dropout_10, ir.getMainGraph());

  DropoutOp *dropout = dynamic_cast<DropoutOp *>(op.get());
  BOOST_CHECK_NE(dropout, nullptr);
  dropout->setSeedModifier(8);
  BOOST_CHECK_EQUAL(dropout->getSeedModifier(), 8);
}
