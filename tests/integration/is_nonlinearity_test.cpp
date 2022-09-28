// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE IsNonLinearityTest

#include <boost/test/unit_test.hpp>
#include <memory>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(IsNonLinearityTest) {
  popart::Ir ir;

  // Is 'div' a non-linearity? - No
  std::unique_ptr<Op> div =
      OpManager::createOp(Onnx::Operators::Div_7, ir.getMainGraph());
  BOOST_CHECK(!div->isElementWiseUnary());

  // Is 'tanh' a non-linearity? - Yes
  std::unique_ptr<Op> tanh =
      OpManager::createOp(Onnx::Operators::Tanh_6, ir.getMainGraph());
  BOOST_CHECK(tanh->isElementWiseUnary());

  // Is 'softmax' a non-linearity? - Yes
  std::unique_ptr<Op> sfm =
      OpManager::createOp(Onnx::Operators::Softmax_1, ir.getMainGraph());
  BOOST_CHECK(sfm->isElementWiseUnary());

  // Is 'relu' a non-linearity? - Yes
  std::unique_ptr<Op> relu =
      OpManager::createOp(Onnx::Operators::Relu_6, ir.getMainGraph());
  BOOST_CHECK(relu->isElementWiseUnary());

  // Is 'sigmoid' a non-linearity? - Yes
  std::unique_ptr<Op> sgm =
      OpManager::createOp(Onnx::Operators::Sigmoid_6, ir.getMainGraph());
  BOOST_CHECK(sgm->isElementWiseUnary());
}
