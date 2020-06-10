// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MatMulTest

#include <boost/test/unit_test.hpp>
#include <onnx/onnx_pb.h>
#include <popart/ir.hpp>
#include <popart/op/matmul.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/op/matmulx.hpp>
#include <popart/tensor.hpp>

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

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph);
  lhs.info.set(popart::DataType::FLOAT, {2, 2});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph);
  rhs.info.set(popart::DataType::FLOAT, {2, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph);
  mm.output->insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.outInfo(0).dim(0) == 2);
  BOOST_CHECK(mm.outInfo(0).dim(1) == 2);
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

      popart::Tensor lhsOut("out", popart::TensorType::ActGrad, graph);
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

      popart::Tensor rhsOut("out", popart::TensorType::ActGrad, graph);
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

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph);
  lhs.info.set(popart::DataType::FLOAT, {3, 2});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph);
  rhs.info.set(popart::DataType::FLOAT, {2, 6});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph);
  mm.output->insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.outInfo(0).dim(0) == 3);
  BOOST_CHECK(mm.outInfo(0).dim(1) == 6);
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

      popart::Tensor lhsOut("out", popart::TensorType::ActGrad, graph);
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

      popart::Tensor rhsOut("out", popart::TensorType::ActGrad, graph);
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

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph);
  lhs.info.set(popart::DataType::FLOAT, {2, 1, 4, 3, 2});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph);
  rhs.info.set(popart::DataType::FLOAT, {3, 1, 2, 6});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph);
  mm.output->insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.outInfo(0).dim(0) == 2);
  BOOST_CHECK(mm.outInfo(0).dim(1) == 3);
  BOOST_CHECK(mm.outInfo(0).dim(2) == 4);
  BOOST_CHECK(mm.outInfo(0).dim(3) == 3);
  BOOST_CHECK(mm.outInfo(0).dim(4) == 6);
  BOOST_CHECK(mm.outInfo(0).dataType() == popart::DataType::FLOAT);
  BOOST_CHECK(mm.rhsIn() == &rhs);
  BOOST_CHECK(mm.lhsIn() == &lhs);

  auto gradOps = mm.getGradOps();
  BOOST_CHECK(gradOps.size() == 2);
  BOOST_CHECK(gradOps[0]->isConvertibleTo<popart::MatMulLhsGradOp>());
  BOOST_CHECK(gradOps[1]->isConvertibleTo<popart::MatMulRhsGradOp>());

  std::vector<popart::Tensor> tensors;
  tensors.reserve(gradOps.size());

  for (auto &op : gradOps) {
    // Danger: Can cause a realloc which invalidates pointers. This won't happen
    // if the vector has space reserved.
    tensors.emplace_back("out", popart::TensorType::ActGrad, graph);
    op->output->reset(0, &tensors.back());
    op->setup();
  }

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

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph);
  lhs.info.set(popart::DataType::FLOAT, {2, 2, 3});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph);
  rhs.info.set(popart::DataType::FLOAT, {2, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph);
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

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph);
  lhs.info.set(popart::DataType::FLOAT, {2, 3});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph);
  rhs.info.set(popart::DataType::FLOAT, {10, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph);
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

  popart::Tensor lhs("lhs", popart::TensorType::ActGrad, graph);
  lhs.info.set(popart::DataType::FLOAT, {});
  mm.input->insert(0, &lhs);

  popart::Tensor rhs("rhs", popart::TensorType::ActGrad, graph);
  rhs.info.set(popart::DataType::FLOAT, {10, 2});
  mm.input->insert(1, &rhs);

  popart::Tensor out("out", popart::TensorType::ActGrad, graph);
  mm.output->insert(0, &out);

  // Test the setup is correct
  BOOST_CHECK_THROW(mm.setup(), error);
}
