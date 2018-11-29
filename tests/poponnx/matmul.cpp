#define BOOST_TEST_MODULE MatMulTest

#include <boost/test/unit_test.hpp>

#include <poponnx/error.hpp>
#include <poponnx/op/matmul.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>

#include <poponnx/popx/op/matmulx.hpp>

using namespace poponnx;

// Test the simple [2x2] * [2x2] matrix
BOOST_AUTO_TEST_CASE(MatMul_Case1) {

  // Setup
  onnx::NodeProto node;
  node.set_op_type("MatMul");

  poponnx::Ir ir;

  poponnx::MatMulOp mm(node, &ir);

  poponnx::Tensor lhs("lhs", poponnx::TensorType::ActGrad, ir);
  lhs.info.set(poponnx::TP::FLOAT, {2, 2});
  mm.input.insert(0, &lhs);

  poponnx::Tensor rhs("rhs", poponnx::TensorType::ActGrad, ir);
  rhs.info.set(poponnx::TP::FLOAT, {2, 2});
  mm.input.insert(1, &rhs);

  poponnx::Tensor out("out", poponnx::TensorType::ActGrad, ir);
  mm.output.insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.output.tensor(0)->info.dim(0) == 2);
  BOOST_CHECK(mm.output.tensor(0)->info.dim(1) == 2);
  BOOST_CHECK(mm.output.tensor(0)->info.dataType() == poponnx::TP::FLOAT);
  BOOST_CHECK(mm.rhsIn() == &rhs);
  BOOST_CHECK(mm.lhsIn() == &lhs);

  // Test the clone operation
  std::unique_ptr<poponnx::Op> mmClone = mm.clone();
  poponnx::MatMulOp *mmPtr = dynamic_cast<poponnx::MatMulOp *>(mmClone.get());
  BOOST_CHECK(mmPtr != nullptr);
  // Clone aka copy does not copy input & output???
  // BOOST_CHECK(mmPtr->output.tensor(0)->info.dim(0) == 2);
  // BOOST_CHECK(mmPtr->output.tensor(0)->info.dim(1) == 2);
  // BOOST_CHECK(mmPtr->output.tensor(0)->info.dataType() ==
  // poponnx::TP::FLOAT); BOOST_CHECK(mmPtr->rhsIn() == &rhs);
  // BOOST_CHECK(mmPtr->lhsIn() == &lhs);

  auto gradOps = mm.getGradOps();
  BOOST_CHECK(gradOps.size() == 2);

  for (auto &&gradOp : gradOps) {
    poponnx::Op *op = gradOp.get();
    if (typeid(*op) == typeid(poponnx::MatMulLhsGradOp)) {
      poponnx::MatMulLhsGradOp *lhsGradOp =
          dynamic_cast<poponnx::MatMulLhsGradOp *>(op);

      poponnx::Tensor lhsOut("out", poponnx::TensorType::ActGrad, ir);
      lhsGradOp->output.insert(0, &lhsOut);

      BOOST_CHECK(lhsGradOp->getGradInputIndex() == 0);
      BOOST_CHECK(lhsGradOp->getRhsInputIndex() == 1);

      lhsGradOp->setup();
      BOOST_CHECK(lhsGradOp->output.tensor(0)->info.dim(0) == 2);
      BOOST_CHECK(lhsGradOp->output.tensor(0)->info.dim(1) == 2);
      BOOST_CHECK(lhsGradOp->output.tensor(0)->info.dataType() ==
                  poponnx::TP::FLOAT);

      auto gradInfo = lhsGradOp->gradInputInfo();
      std::vector<GradInOutMapper> expectedGradInfo = {
          {0, 0, poponnx::GradOpInType::GRADOUT},
          {1, 1, poponnx::GradOpInType::IN}};
      BOOST_CHECK(gradInfo == expectedGradInfo);

      auto mapping                       = lhsGradOp->gradOutToNonGradIn();
      std::map<int, int> expectedMapping = {{0, 0}};
      BOOST_CHECK(mapping == expectedMapping);
    } else if (typeid(*op) == typeid(poponnx::MatMulRhsGradOp)) {
      poponnx::MatMulRhsGradOp *rhsGradOp =
          dynamic_cast<poponnx::MatMulRhsGradOp *>(op);

      poponnx::Tensor rhsOut("out", poponnx::TensorType::ActGrad, ir);
      rhsGradOp->output.insert(0, &rhsOut);

      BOOST_CHECK(rhsGradOp->getGradInputIndex() == 0);
      BOOST_CHECK(rhsGradOp->getLhsInputIndex() == 1);

      rhsGradOp->setup();
      BOOST_CHECK(rhsGradOp->output.tensor(0)->info.dim(0) == 2);
      BOOST_CHECK(rhsGradOp->output.tensor(0)->info.dim(1) == 2);
      BOOST_CHECK(rhsGradOp->output.tensor(0)->info.dataType() ==
                  poponnx::TP::FLOAT);

      auto gradInfo = rhsGradOp->gradInputInfo();
      std::vector<GradInOutMapper> expectedGradInfo = {
          {0, 0, poponnx::GradOpInType::GRADOUT},
          {1, 0, poponnx::GradOpInType::IN}};

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
  onnx::NodeProto node;
  node.set_op_type("MatMul");

  poponnx::Ir ir;

  poponnx::MatMulOp mm(node, &ir);

  poponnx::Tensor lhs("lhs", poponnx::TensorType::ActGrad, ir);
  lhs.info.set(poponnx::TP::FLOAT, {3, 2});
  mm.input.insert(0, &lhs);

  poponnx::Tensor rhs("rhs", poponnx::TensorType::ActGrad, ir);
  rhs.info.set(poponnx::TP::FLOAT, {2, 6});
  mm.input.insert(1, &rhs);

  poponnx::Tensor out("out", poponnx::TensorType::ActGrad, ir);
  mm.output.insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.output.tensor(0)->info.dim(0) == 3);
  BOOST_CHECK(mm.output.tensor(0)->info.dim(1) == 6);
  BOOST_CHECK(mm.output.tensor(0)->info.dataType() == poponnx::TP::FLOAT);
  BOOST_CHECK(mm.rhsIn() == &rhs);
  BOOST_CHECK(mm.lhsIn() == &lhs);

  // Test the clone operation
  std::unique_ptr<poponnx::Op> mmClone = mm.clone();
  poponnx::MatMulOp *mmPtr = dynamic_cast<poponnx::MatMulOp *>(mmClone.get());
  BOOST_CHECK(mmPtr != nullptr);
  // Clone aka copy does not copy input & output???
  // BOOST_CHECK(mmPtr->output.tensor(0)->info.dim(0) == 2);
  // BOOST_CHECK(mmPtr->output.tensor(0)->info.dim(1) == 2);
  // BOOST_CHECK(mmPtr->output.tensor(0)->info.dataType() ==
  // poponnx::TP::FLOAT); BOOST_CHECK(mmPtr->rhsIn() == &rhs);
  // BOOST_CHECK(mmPtr->lhsIn() == &lhs);

  auto gradOps = mm.getGradOps();
  BOOST_CHECK(gradOps.size() == 2);

  for (auto &&gradOp : gradOps) {
    poponnx::Op *op = gradOp.get();
    if (typeid(*op) == typeid(poponnx::MatMulLhsGradOp)) {
      poponnx::MatMulLhsGradOp *lhsGradOp =
          dynamic_cast<poponnx::MatMulLhsGradOp *>(op);

      poponnx::Tensor lhsOut("out", poponnx::TensorType::ActGrad, ir);
      lhsGradOp->output.insert(0, &lhsOut);

      BOOST_CHECK(lhsGradOp->getGradInputIndex() == 0);
      BOOST_CHECK(lhsGradOp->getRhsInputIndex() == 1);

      lhsGradOp->setup();
      BOOST_CHECK(lhsGradOp->output.tensor(0)->info.dim(0) == 3);
      BOOST_CHECK(lhsGradOp->output.tensor(0)->info.dim(1) == 2);
      BOOST_CHECK(lhsGradOp->output.tensor(0)->info.dataType() ==
                  poponnx::TP::FLOAT);

      auto gradInfo = lhsGradOp->gradInputInfo();
      std::vector<GradInOutMapper> expectedGradInfo = {
          {0, 0, poponnx::GradOpInType::GRADOUT},
          {1, 1, poponnx::GradOpInType::IN}};
      BOOST_CHECK(gradInfo == expectedGradInfo);

      auto mapping                       = lhsGradOp->gradOutToNonGradIn();
      std::map<int, int> expectedMapping = {{0, 0}};
      BOOST_CHECK(mapping == expectedMapping);
    } else if (typeid(*op) == typeid(poponnx::MatMulRhsGradOp)) {
      poponnx::MatMulRhsGradOp *rhsGradOp =
          dynamic_cast<poponnx::MatMulRhsGradOp *>(op);

      poponnx::Tensor rhsOut("out", poponnx::TensorType::ActGrad, ir);
      rhsGradOp->output.insert(0, &rhsOut);

      BOOST_CHECK(rhsGradOp->getGradInputIndex() == 0);
      BOOST_CHECK(rhsGradOp->getLhsInputIndex() == 1);

      rhsGradOp->setup();
      BOOST_CHECK(rhsGradOp->output.tensor(0)->info.dim(0) == 2);
      BOOST_CHECK(rhsGradOp->output.tensor(0)->info.dim(1) == 6);
      BOOST_CHECK(rhsGradOp->output.tensor(0)->info.dataType() ==
                  poponnx::TP::FLOAT);

      auto gradInfo = rhsGradOp->gradInputInfo();
      std::vector<GradInOutMapper> expectedGradInfo = {
          {0, 0, poponnx::GradOpInType::GRADOUT},
          {1, 0, poponnx::GradOpInType::IN}};

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
  onnx::NodeProto node;
  node.set_op_type("MatMul");

  poponnx::Ir ir;

  poponnx::MatMulOp mm(node, &ir);

  poponnx::Tensor lhs("lhs", poponnx::TensorType::ActGrad, ir);
  lhs.info.set(poponnx::TP::FLOAT, {2, 1, 4, 3, 2});
  mm.input.insert(0, &lhs);

  poponnx::Tensor rhs("rhs", poponnx::TensorType::ActGrad, ir);
  rhs.info.set(poponnx::TP::FLOAT, {3, 1, 2, 6});
  mm.input.insert(1, &rhs);

  poponnx::Tensor out("out", poponnx::TensorType::ActGrad, ir);
  mm.output.insert(0, &out);

  // Test the setup is correct
  mm.setup();
  BOOST_CHECK(mm.output.tensor(0)->info.dim(0) == 2);
  BOOST_CHECK(mm.output.tensor(0)->info.dim(1) == 3);
  BOOST_CHECK(mm.output.tensor(0)->info.dim(2) == 4);
  BOOST_CHECK(mm.output.tensor(0)->info.dim(3) == 3);
  BOOST_CHECK(mm.output.tensor(0)->info.dim(4) == 6);
  BOOST_CHECK(mm.output.tensor(0)->info.dataType() == poponnx::TP::FLOAT);
  BOOST_CHECK(mm.rhsIn() == &rhs);
  BOOST_CHECK(mm.lhsIn() == &lhs);

  auto gradOps = mm.getGradOps();
  BOOST_CHECK(gradOps.size() == 2);
  BOOST_CHECK(gradOps[0]->isConvertibleTo<poponnx::MatMulLhsGradOp>());
  BOOST_CHECK(gradOps[1]->isConvertibleTo<poponnx::MatMulRhsGradOp>());

  std::vector<poponnx::Tensor> tensors;
  tensors.reserve(gradOps.size());

  for (auto &op : gradOps) {
    // Danger: Can cause a realloc which invalidates pointers. This won't happen
    // if the vector has space reserved.
    tensors.emplace_back("out", poponnx::TensorType::ActGrad, ir);
    op->output.reset(0, &tensors.back());
    op->setup();
  }

  BOOST_CHECK(gradOps[0]->output.tensor(0)->info ==
              mm.input.tensor(poponnx::MatMulOp::getLhsInputIndex())->info);
  BOOST_CHECK(gradOps[1]->output.tensor(0)->info ==
              mm.input.tensor(poponnx::MatMulOp::getRhsInputIndex())->info);
}

// Test invalid rank on lhs
BOOST_AUTO_TEST_CASE(MatMul_ErrorCase1) {

  // Setup
  onnx::NodeProto node;
  node.set_op_type("MatMul");

  poponnx::Ir ir;

  poponnx::MatMulOp mm(node, &ir);

  poponnx::Tensor lhs("lhs", poponnx::TensorType::ActGrad, ir);
  lhs.info.set(poponnx::TP::FLOAT, {2, 2, 3});
  mm.input.insert(0, &lhs);

  poponnx::Tensor rhs("rhs", poponnx::TensorType::ActGrad, ir);
  rhs.info.set(poponnx::TP::FLOAT, {2, 2});
  mm.input.insert(1, &rhs);

  poponnx::Tensor out("out", poponnx::TensorType::ActGrad, ir);
  mm.output.insert(0, &out);

  // Test the setup is correct
  BOOST_CHECK_THROW(mm.setup(), error);
}

// Test invalid matrix multiplication
BOOST_AUTO_TEST_CASE(MatMul_ErrorCase3) {

  // Setup
  onnx::NodeProto node;
  node.set_op_type("MatMul");

  poponnx::Ir ir;

  poponnx::MatMulOp mm(node, &ir);

  poponnx::Tensor lhs("lhs", poponnx::TensorType::ActGrad, ir);
  lhs.info.set(poponnx::TP::FLOAT, {2, 3});
  mm.input.insert(0, &lhs);

  poponnx::Tensor rhs("rhs", poponnx::TensorType::ActGrad, ir);
  rhs.info.set(poponnx::TP::FLOAT, {10, 2});
  mm.input.insert(1, &rhs);

  poponnx::Tensor out("out", poponnx::TensorType::ActGrad, ir);
  mm.output.insert(0, &out);

  // Test the setup is correct
  BOOST_CHECK_THROW(mm.setup(), error);
}

// Test invalid matrix multiplication
BOOST_AUTO_TEST_CASE(MatMul_ErrorCase4) {

  // Setup
  onnx::NodeProto node;
  node.set_op_type("MatMul");

  poponnx::Ir ir;

  poponnx::MatMulOp mm(node, &ir);

  poponnx::Tensor lhs("lhs", poponnx::TensorType::ActGrad, ir);
  lhs.info.set(poponnx::TP::FLOAT, {});
  mm.input.insert(0, &lhs);

  poponnx::Tensor rhs("rhs", poponnx::TensorType::ActGrad, ir);
  rhs.info.set(poponnx::TP::FLOAT, {10, 2});
  mm.input.insert(1, &rhs);

  poponnx::Tensor out("out", poponnx::TensorType::ActGrad, ir);
  mm.output.insert(0, &out);

  // Test the setup is correct
  BOOST_CHECK_THROW(mm.setup(), error);
}
