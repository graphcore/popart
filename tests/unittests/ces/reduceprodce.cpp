// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprReduceProdUnittest

#include <boost/optional.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/ces/reduceprodce.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/reduceprod.hpp>

#include <string>

using namespace popart;

const Shape SHAPE          = {2, 3, 5};
const int64_t RANK         = SHAPE.size();
const int64_t ELEMENT_SIZE = sizeof(int64_t);

// Constant folds a ReduceProd with the provided keepdims and axes atributes and
// a 2 * 3 * 5 constant tensor of 2s as input.
void check_combination(nonstd::optional_lite::optional<Shape> axes,
                       int64_t keepdims,
                       int64_t num_elements) {

  // Print out axes and keepdims
  BOOST_TEST_MESSAGE("keepdims: " + std::to_string(keepdims));
  std::string axes_string;
  if (axes.has_value()) {
    for (auto axis : axes.value()) {
      axes_string += std::to_string(axis) + " ";
    }
    BOOST_TEST_MESSAGE("axes: " + axes_string);
  } else
    BOOST_TEST_MESSAGE("no axes provided");
  BOOST_TEST_MESSAGE("");

  // Create a small graph with one node and one constant tensor
  Ir ir;
  Graph &g = ir.getMainGraph();

  Op::Settings settings(g, "test_reduceprod");

  // set up input and output tensors
  const TensorInfo input_info{DataType::INT64, SHAPE};
  const std::vector<int64_t> input_data(input_info.nelms(), 2);
  g.addConstInit("input", input_info, input_data.data(), "input_init");
  g.addActGrad("output");

  auto op = g.createConnectedOp<ReduceProdOp>(
      {{ReduceProdOp::getInIndex(), "input"}},
      {{ReduceProdOp::getOutIndex(), "output"}},
      Onnx::Operators::ReduceProd_11,
      axes,
      keepdims,
      settings);

  ConstExprReduceProd ce_reduce_prod(op);

  int64_t expected_element = 1
                             << (SHAPE[0] * SHAPE[1] * SHAPE[2] / num_elements);

  // The result returned is just a memcpy of the raw array in the resulting
  // tensor. 8 bytes per element.
  auto result = ce_reduce_prod.compute();

  // Check that the result has the size we expect
  BOOST_REQUIRE_MESSAGE(num_elements * ELEMENT_SIZE == result.size(),
                        "Number of returned bytes was " +
                            std::to_string(result.size()) + "; expected " +
                            std::to_string(num_elements * ELEMENT_SIZE));

  // Compare the bytes to what we expect to see
  for (int i = 0; i < num_elements; i++) {
    int64_t returned_element;
    std::memcpy(
        &returned_element, &result.data()[i * ELEMENT_SIZE], ELEMENT_SIZE);

    BOOST_CHECK_MESSAGE(expected_element == returned_element,
                        "expected element " + std::to_string(expected_element) +
                            ", got " + std::to_string(returned_element));
  }
}

BOOST_AUTO_TEST_CASE(TestConstExprReduceProd) {
  // positive axes
  for (int combination = 0; combination < (1 << RANK); ++combination) {
    Shape axes;
    int64_t num_elements = 1;
    for (int i = 0; i < RANK; ++i) {
      if ((combination >> i) & 1)
        axes.push_back(i);
      else
        num_elements *= SHAPE[i];
    }
    check_combination(axes, 0, num_elements);
    check_combination(axes, 1, num_elements);
  }

  // negative axes
  for (int combination = 1; combination < (1 << RANK); ++combination) {
    Shape axes;
    int64_t num_elements = 1;
    for (int i = 0; i < RANK; i++) {
      if ((combination >> i) & 1)
        axes.push_back(-i - 1);
      else
        num_elements *= SHAPE[2 - i];
    }
    check_combination(axes, 0, num_elements);
    check_combination(axes, 1, num_elements);
  }

  // no axes
  check_combination(nonstd::nullopt, 0, 1);
  check_combination(nonstd::nullopt, 1, 1);

  // duplicated axes
  check_combination(Shape{1, 1}, 0, 10);
  check_combination(Shape{1, -2}, 1, 10);

  // mixed negative and positive axes
  check_combination(Shape{1, -1}, 0, 2);
  check_combination(Shape{1, -3}, 1, 5);
}