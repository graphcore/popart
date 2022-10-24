// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MatMulTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <map>
#include <memory>
#include <typeinfo>
#include <vector>
#include <popart/ir.hpp>
#include <popart/op/matmul.hpp>
#include <popart/tensor.hpp>

#include "popart/datatype.hpp"
#include "popart/debugcontext.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
class error;
} // namespace popart

using namespace popart;

// Test the simple [2x2] * [2x2] matrix
BOOST_AUTO_TEST_CASE(MatMul_Case1) {

  // Setup

  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType());

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT, {2, 2});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT, {2, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);
  mm.output->insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.outInfo(0).dim(0) == 2);
  BOOST_CHECK(mm.outInfo(0).dim(1) == 2);
  BOOST_CHECK(!mm.isPow2ScaledMatMul());
  BOOST_CHECK(mm.outInfo(0).dataType() == popart::DataType::FLOAT);
  BOOST_CHECK(mm.rhsIn() == &rhs);
  BOOST_CHECK(mm.lhsIn() == &lhs);

  // Test the clone operation
  std::unique_ptr<popart::Op> mmClone = mm.clone();
  popart::MatMulOp *mmPtr = dynamic_cast<popart::MatMulOp *>(mmClone.get());
  BOOST_CHECK(mmPtr != nullptr);
  // Clone aka copy does not copy input & output???
  // BOOST_CHECK(mmPtr->outInfo(0).dim(0) == 2);
  // BOOST_CHECK(mmPtr->outInfo(0).dim(1) == 2);
  // BOOST_CHECK(mmPtr->outInfo(0).dataType() ==
  // popart::DataType::FLOAT); BOOST_CHECK(mmPtr->rhsIn() == &rhs);
  // BOOST_CHECK(mmPtr->lhsIn() == &lhs);

  auto gradOps = mm.getGradOps();
  BOOST_CHECK(gradOps.size() == 2);

  for (auto &&gradOp : gradOps) {
    popart::Op *op = gradOp.get();
    if (typeid(*op) == typeid(popart::MatMulLhsGradOp)) {
      popart::MatMulLhsGradOp *lhsGradOp =
          dynamic_cast<popart::MatMulLhsGradOp *>(op);

      popart::Tensor lhsOut("out", popart::TensorType::ActGrad, graph, di);
      lhsGradOp->output->insert(0, &lhsOut);

      BOOST_CHECK(lhsGradOp->getGradInIndex() == 0);
      BOOST_CHECK(lhsGradOp->getRhsInIndex() == 1);

      lhsGradOp->setup();
      BOOST_CHECK(lhsGradOp->outInfo(0).dim(0) == 2);
      BOOST_CHECK(lhsGradOp->outInfo(0).dim(1) == 2);
      BOOST_CHECK(lhsGradOp->outInfo(0).dataType() == popart::DataType::FLOAT);

      auto gradInfo = lhsGradOp->gradInputInfo();
      std::vector<GradInOutMapper> expectedGradInfo = {
          {0, 0, popart::GradOpInType::GradOut},
          {1, 1, popart::GradOpInType::In}};
      BOOST_CHECK(gradInfo == expectedGradInfo);

      auto mapping                       = lhsGradOp->gradOutToNonGradIn();
      std::map<int, int> expectedMapping = {{0, 0}};
      BOOST_CHECK(mapping == expectedMapping);
    } else if (typeid(*op) == typeid(popart::MatMulRhsGradOp)) {
      popart::MatMulRhsGradOp *rhsGradOp =
          dynamic_cast<popart::MatMulRhsGradOp *>(op);

      popart::Tensor rhsOut("out", popart::TensorType::ActGrad, graph, di);
      rhsGradOp->output->insert(0, &rhsOut);

      BOOST_CHECK(rhsGradOp->getGradInIndex() == 0);
      BOOST_CHECK(rhsGradOp->getLhsInIndex() == 1);

      rhsGradOp->setup();
      BOOST_CHECK(rhsGradOp->outInfo(0).dim(0) == 2);
      BOOST_CHECK(rhsGradOp->outInfo(0).dim(1) == 2);
      BOOST_CHECK(rhsGradOp->outInfo(0).dataType() == popart::DataType::FLOAT);

      auto gradInfo = rhsGradOp->gradInputInfo();
      std::vector<GradInOutMapper> expectedGradInfo = {
          {0, 0, popart::GradOpInType::GradOut},
          {1, 0, popart::GradOpInType::In}};

      BOOST_CHECK(gradInfo == expectedGradInfo);

      auto mapping                       = rhsGradOp->gradOutToNonGradIn();
      std::map<int, int> expectedMapping = {{0, 1}};
      BOOST_CHECK(mapping == expectedMapping);

    } else {
      // Unexpected grad op
      BOOST_CHECK(false);
    }
  }
}

// Test the simple [3x2] * [2x6] matrix
BOOST_AUTO_TEST_CASE(MatMul_Case2) {

  // Setup

  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType());

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT, {3, 2});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT, {2, 6});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);
  mm.output->insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.outInfo(0).dim(0) == 3);
  BOOST_CHECK(mm.outInfo(0).dim(1) == 6);
  BOOST_CHECK(!mm.isPow2ScaledMatMul());
  BOOST_CHECK(mm.outInfo(0).dataType() == popart::DataType::FLOAT);
  BOOST_CHECK(mm.rhsIn() == &rhs);
  BOOST_CHECK(mm.lhsIn() == &lhs);

  // Test the clone operation
  std::unique_ptr<popart::Op> mmClone = mm.clone();
  popart::MatMulOp *mmPtr = dynamic_cast<popart::MatMulOp *>(mmClone.get());
  BOOST_CHECK(mmPtr != nullptr);
  // Clone aka copy does not copy input & output???
  // BOOST_CHECK(mmPtr->outInfo(0).dim(0) == 2);
  // BOOST_CHECK(mmPtr->outInfo(0).dim(1) == 2);
  // BOOST_CHECK(mmPtr->outInfo(0).dataType() ==
  // popart::DataType::FLOAT); BOOST_CHECK(mmPtr->rhsIn() == &rhs);
  // BOOST_CHECK(mmPtr->lhsIn() == &lhs);

  auto gradOps = mm.getGradOps();
  BOOST_CHECK(gradOps.size() == 2);

  for (auto &&gradOp : gradOps) {
    popart::Op *op = gradOp.get();
    if (typeid(*op) == typeid(popart::MatMulLhsGradOp)) {
      popart::MatMulLhsGradOp *lhsGradOp =
          dynamic_cast<popart::MatMulLhsGradOp *>(op);

      popart::Tensor lhsOut("out", popart::TensorType::ActGrad, graph, di);
      lhsGradOp->output->insert(0, &lhsOut);

      BOOST_CHECK(lhsGradOp->getGradInIndex() == 0);
      BOOST_CHECK(lhsGradOp->getRhsInIndex() == 1);

      lhsGradOp->setup();
      BOOST_CHECK(lhsGradOp->outInfo(0).dim(0) == 3);
      BOOST_CHECK(lhsGradOp->outInfo(0).dim(1) == 2);
      BOOST_CHECK(lhsGradOp->outInfo(0).dataType() == popart::DataType::FLOAT);

      auto gradInfo = lhsGradOp->gradInputInfo();
      std::vector<GradInOutMapper> expectedGradInfo = {
          {0, 0, popart::GradOpInType::GradOut},
          {1, 1, popart::GradOpInType::In}};
      BOOST_CHECK(gradInfo == expectedGradInfo);

      auto mapping                       = lhsGradOp->gradOutToNonGradIn();
      std::map<int, int> expectedMapping = {{0, 0}};
      BOOST_CHECK(mapping == expectedMapping);
    } else if (typeid(*op) == typeid(popart::MatMulRhsGradOp)) {
      popart::MatMulRhsGradOp *rhsGradOp =
          dynamic_cast<popart::MatMulRhsGradOp *>(op);

      popart::Tensor rhsOut("out", popart::TensorType::ActGrad, graph, di);
      rhsGradOp->output->insert(0, &rhsOut);

      BOOST_CHECK(rhsGradOp->getGradInIndex() == 0);
      BOOST_CHECK(rhsGradOp->getLhsInIndex() == 1);

      rhsGradOp->setup();
      BOOST_CHECK(rhsGradOp->outInfo(0).dim(0) == 2);
      BOOST_CHECK(rhsGradOp->outInfo(0).dim(1) == 6);
      BOOST_CHECK(rhsGradOp->outInfo(0).dataType() == popart::DataType::FLOAT);

      auto gradInfo = rhsGradOp->gradInputInfo();
      std::vector<GradInOutMapper> expectedGradInfo = {
          {0, 0, popart::GradOpInType::GradOut},
          {1, 0, popart::GradOpInType::In}};

      BOOST_CHECK(gradInfo == expectedGradInfo);

      auto mapping                       = rhsGradOp->gradOutToNonGradIn();
      std::map<int, int> expectedMapping = {{0, 1}};
      BOOST_CHECK(mapping == expectedMapping);

    } else {
      // Unexpected grad op
      BOOST_CHECK(false);
    }
  }
}

// Test the simple [2x1x4x3x2] * [3x1x2x6] matrix
BOOST_AUTO_TEST_CASE(MatMul_Case3) {
  // Setup

  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType());

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT, {2, 1, 4, 3, 2});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT, {3, 1, 2, 6});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);
  mm.output->insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.outInfo(0).dim(0) == 2);
  BOOST_CHECK(mm.outInfo(0).dim(1) == 3);
  BOOST_CHECK(mm.outInfo(0).dim(2) == 4);
  BOOST_CHECK(mm.outInfo(0).dim(3) == 3);
  BOOST_CHECK(mm.outInfo(0).dim(4) == 6);
  BOOST_CHECK(!mm.isPow2ScaledMatMul());
  BOOST_CHECK(mm.outInfo(0).dataType() == popart::DataType::FLOAT);
  BOOST_CHECK(mm.rhsIn() == &rhs);
  BOOST_CHECK(mm.lhsIn() == &lhs);

  auto gradOps = mm.getGradOps();
  BOOST_CHECK(gradOps.size() == 2);
  BOOST_CHECK(gradOps[0]->isConvertibleTo<popart::MatMulLhsGradOp>());
  BOOST_CHECK(gradOps[1]->isConvertibleTo<popart::MatMulRhsGradOp>());

  popart::Tensor t1("out", popart::TensorType::ActGrad, graph, di);
  gradOps[0]->output->reset(0, &t1);
  gradOps[0]->setup();

  popart::Tensor t2("out", popart::TensorType::ActGrad, graph, di);
  gradOps[1]->output->reset(0, &t2);
  gradOps[1]->setup();

  BOOST_CHECK(gradOps[0]->output->tensor(0)->info ==
              mm.input->tensor(popart::MatMulOp::getLhsInIndex())->info);
  BOOST_CHECK(gradOps[1]->output->tensor(0)->info ==
              mm.input->tensor(popart::MatMulOp::getRhsInIndex())->info);
}

// Test invalid rank on lhs
BOOST_AUTO_TEST_CASE(MatMul_ErrorCase1) {

  // Setup

  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType());

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT, {2, 2, 3});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT, {2, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);
  mm.output->insert(0, &out);

  // Test the setup is correct
  BOOST_CHECK_THROW(mm.setup(), error);
}

// Test invalid matrix multiplication
BOOST_AUTO_TEST_CASE(MatMul_ErrorCase3) {

  // Setup

  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType());

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT, {2, 3});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT, {10, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);
  mm.output->insert(0, &out);

  // Test the setup is correct
  BOOST_CHECK_THROW(mm.setup(), error);
}

// Test invalid matrix multiplication
BOOST_AUTO_TEST_CASE(MatMul_ErrorCase4) {

  // Setup

  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType());

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT, {});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT, {10, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);
  mm.output->insert(0, &out);

  // Test the setup is correct
  BOOST_CHECK_THROW(mm.setup(), error);
}

BOOST_AUTO_TEST_CASE(MatMul_Float8ErrorCases) {
  // Setup
  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType(),
                      MatMulPartialsType::HALF);

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  // Mixing FP8 with any other non-FP8 type throws
  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT8_143, {2, 10});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT16, {10, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);
  mm.output->insert(0, &out);

  std::string expectedMsg;
  auto errorMessageMatches = [&expectedMsg](popart::error const &error) {
    return std::string(error.what()).find(expectedMsg) != std::string::npos;
  };

  expectedMsg = "Invalid combination of operand types";
  BOOST_CHECK(!mm.isPow2ScaledMatMul());
  BOOST_CHECK_EXCEPTION(mm.setup(), error, errorMessageMatches);

  // Not providing log2 scale for two valid FP8 inputs throws
  mm.input->clear();
  rhs.info.set(popart::DataType::FLOAT8_143, {10, 2});

  mm.input->insert(0, &lhs);
  mm.input->insert(1, &rhs);

  expectedMsg = "Log2 scale input must be provided";
  BOOST_CHECK(!mm.isPow2ScaledMatMul());
  BOOST_CHECK_EXCEPTION(mm.setup(), error, errorMessageMatches);

  // Providing log2 scale for non-FP8 input tensors throws.
  popart::Tensor lhsInvalid(
      "lhsInvalid", popart::TensorType::ActGrad, graph, di);
  lhsInvalid.info.set(popart::DataType::FLOAT, {2, 2});
  popart::Tensor rhsInvalid(
      "lhsInvalid", popart::TensorType::ActGrad, graph, di);
  rhsInvalid.info.set(popart::DataType::FLOAT, {2, 2});

  popart::Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, graph, di);
  log2Scale.info.set(popart::DataType::INT32, {});

  mm.input->clear();
  mm.input->insert(0, &lhsInvalid);
  mm.input->insert(1, &rhsInvalid);
  mm.input->insert(2, &log2Scale);

  expectedMsg = "Log2 scale input not accepted";
  BOOST_CHECK(!mm.isPow2ScaledMatMul());
  BOOST_CHECK_EXCEPTION(mm.setup(), error, errorMessageMatches);

  // Providing a non int32 log 2 scale tensor for FP8 inputs throws.
  popart::Tensor invalidLog2Scale(
      "invalidLogScale", popart::TensorType::ActGrad, graph, di);
  invalidLog2Scale.info.set(popart::DataType::FLOAT, {});

  mm.input->clear();
  mm.input->insert(0, &lhs);
  mm.input->insert(1, &rhs);
  mm.input->insert(2, &invalidLog2Scale);

  expectedMsg = "Invalid log2 scale input type";
  BOOST_CHECK(!mm.isPow2ScaledMatMul());
  BOOST_CHECK_EXCEPTION(mm.setup(), error, errorMessageMatches);

  // Non-scalar log2 scale tensor throws.
  popart::Tensor nonScalarLog2Scale(
      "nonScalarLog2Scale", popart::TensorType::ActGrad, graph, di);
  nonScalarLog2Scale.info.set(popart::DataType::INT32, {1, 2, 3});

  mm.input->clear();
  mm.input->insert(0, &lhs);
  mm.input->insert(1, &rhs);

  mm.input->insert(2, &nonScalarLog2Scale);
  expectedMsg = "must be a scalar tensor";
  BOOST_CHECK(!mm.isPow2ScaledMatMul());
  BOOST_CHECK_EXCEPTION(mm.setup(), error, errorMessageMatches);

  // Providing an output type that isn't FP16 throws.
  popart::MatMulOp mmOutputType(Onnx::Operators::MatMul_9,
                                {graph, ""},
                                nonstd::nullopt,
                                {},
                                DataType::FLOAT,
                                MatMulPartialsType::HALF);

  mmOutputType.input->insert(0, &lhs);
  mmOutputType.input->insert(1, &rhs);
  mmOutputType.input->insert(2, &log2Scale);

  expectedMsg = "Invalid output type";
  BOOST_CHECK_EXCEPTION(mmOutputType.setup(), error, errorMessageMatches);

  // Providing a partials type that isn't FP16 throws.
  popart::MatMulOp mmPartialsType(Onnx::Operators::MatMul_9,
                                  {graph, ""},
                                  nonstd::nullopt,
                                  {},
                                  DataType::FLOAT16,
                                  MatMulPartialsType::FLOAT);

  mmPartialsType.input->insert(0, &lhs);
  mmPartialsType.input->insert(1, &rhs);
  mmPartialsType.input->insert(2, &log2Scale);

  expectedMsg = "Invalid partials type";
  BOOST_CHECK_EXCEPTION(mmPartialsType.setup(), error, errorMessageMatches);
}

BOOST_AUTO_TEST_CASE(Matmul_Float8) {
  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType(),
                      MatMulPartialsType::HALF);

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT8_143, {2, 10});

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT8_152, {10, 2});

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);

  popart::Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, graph, di);
  log2Scale.info.set(popart::DataType::INT32, {});

  mm.input->insert(0, &lhs);
  mm.input->insert(1, &rhs);
  mm.input->insert(2, &log2Scale);

  mm.output->insert(0, &out);

  // Setup needs to be called to set partials type and outInfo, otherwise
  // segfault will happen
  mm.setup();
  BOOST_CHECK(mm.isPow2ScaledMatMul());
  BOOST_CHECK(mm.outInfo(0).dataType() == DataType::FLOAT16);
  BOOST_CHECK(mm.getPartialsType() == MatMulPartialsType::HALF);
  BOOST_CHECK(mm.hasInput(mm.getLhsInIndex()));
  BOOST_CHECK(mm.hasInput(mm.getRhsInIndex()));
  BOOST_CHECK(mm.hasInput(mm.getLog2ScaleInIndex()));
}

BOOST_AUTO_TEST_CASE(Matmul_RaiseWhenUsingFloat8MatmulInBackwardsPass) {
  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  popart::MatMulOp mm(Onnx::Operators::MatMul_9,
                      {graph, ""},
                      nonstd::nullopt,
                      {},
                      OptionalDataType(),
                      MatMulPartialsType::HALF);

  popart::DebugContext dc;
  popart::DebugInfo di(dc, "test");

  // Mixing FP8 with any other non-FP8 type throws
  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph, di);
  lhs.info.set(popart::DataType::FLOAT8_143, {2, 10});

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph, di);
  rhs.info.set(popart::DataType::FLOAT8_143, {10, 2});

  popart::Tensor log2scale("log2scale", popart::TensorType::ActGrad, graph, di);
  log2scale.info.set(popart::DataType::INT32, {});

  popart::Tensor out("out", popart::TensorType::ActGrad, graph, di);

  mm.input->insert(0, &lhs);
  mm.input->insert(1, &rhs);
  mm.input->insert(2, &log2scale);
  mm.output->insert(0, &out);

  mm.setup();

  // FP8 matmul during training is currently not supported, should
  // not be able to get grad ops when we know the Op is an FP8 matmul
  BOOST_CHECK(mm.isPow2ScaledMatMul());
  BOOST_CHECK_THROW(mm.getGradOps(), error);
}
