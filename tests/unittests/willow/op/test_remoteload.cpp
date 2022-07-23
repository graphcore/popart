// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Willow_Op_RemoteLoad
#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/graphcoreoperators.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/remote.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestGetInplaceVariantReturnsCorrectOpIdentifier) {
  Ir ir;
  auto &mg = ir.getMainGraph();

  auto op = mg.createOp<RemoteLoadOp>(Onnx::CustomOperators::RemoteLoad,
                                      Op::Settings{mg, "RemoteLoad"},
                                      RemoteBufferId{});

  auto inplaceOp =
      op->getInplaceVariant(Onnx::CustomOperators::RemoteLoadInplace);

  // Returns inplace op with the correct opid
  BOOST_REQUIRE_EQUAL(inplaceOp->opid,
                      Onnx::CustomOperators::RemoteLoadInplace);

  // Creating inplace op with any other opid throws
  BOOST_REQUIRE_THROW(op->getInplaceVariant(Onnx::CustomOperators::RemoteLoad),
                      popart::error);
}
