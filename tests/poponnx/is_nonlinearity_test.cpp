#define BOOST_TEST_MODULE IsNonLinearityTest

#include <boost/test/unit_test.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/opmanager.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(IsNonLinearityTest) {
  poponnx::Ir ir;

  // Is 'div' a non-linearity? - No
  std::unique_ptr<Op> div =
      OpManager::createOp(Onnx::Operators::Div_7, ir.getMainGraph());
  BOOST_CHECK(!div.get()->isElementWiseUnary());

  // Is 'tanh' a non-linearity? - Yes
  std::unique_ptr<Op> tanh =
      OpManager::createOp(Onnx::Operators::Tanh_6, ir.getMainGraph());
  BOOST_CHECK(tanh.get()->isElementWiseUnary());

  // Is 'softmax' a non-linearity? - Yes
  std::unique_ptr<Op> sfm =
      OpManager::createOp(Onnx::Operators::Softmax_1, ir.getMainGraph());
  BOOST_CHECK(sfm.get()->isElementWiseUnary());

  // Is 'relu' a non-linearity? - Yes
  std::unique_ptr<Op> relu =
      OpManager::createOp(Onnx::Operators::Relu_6, ir.getMainGraph());
  BOOST_CHECK(relu.get()->isElementWiseUnary());

  // Is 'sigmoid' a non-linearity? - Yes
  std::unique_ptr<Op> sgm =
      OpManager::createOp(Onnx::Operators::Sigmoid_6, ir.getMainGraph());
  BOOST_CHECK(sgm.get()->isElementWiseUnary());
}
