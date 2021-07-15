// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NumpyBroadcastShape

#include <sstream>

#include <boost/test/unit_test.hpp>
#include <popart/error.hpp>
#include <popart/tensorinfo.hpp>

struct BroadcastTestCase {
  std::vector<int64_t> a_shape;
  std::vector<int64_t> b_shape;
  std::vector<int64_t> result_shape;
};

// clang-format off
static const BroadcastTestCase test_cases[] = {
    // Test cases taken from
    // https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
    {{   256, 256, 3}, {       3}, {   256, 256, 3}},
    {{8,   1,   6, 1}, {7,  1, 5}, {8,   7,   6, 5}},
    {{          5, 4}, {       1}, {          5, 4}},
    {{          5, 4}, {       4}, {          5, 4}},
    {{    15,   3, 5}, {15, 1, 5}, {    15,   3, 5}},
    {{    15,   3, 5}, {    3, 5}, {    15,   3, 5}},
    {{    15,   3, 5}, {    3, 1}, {    15,   3, 5}},

    // Test cases taken from
    // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    {{2, 3, 4, 5}, {/* scalar */}, {2, 3, 4, 5}},
    {{2, 3, 4, 5}, {           5}, {2, 3, 4, 5}},
    {{      4, 5}, {  2, 3, 4, 5}, {2, 3, 4, 5}},
    {{   1, 4, 5}, {  2, 3, 1, 1}, {2, 3, 4, 5}},
    {{   3, 4, 5}, {  2, 1, 1, 1}, {2, 3, 4, 5}}
};
// clang-format on

BOOST_AUTO_TEST_CASE(NumpyBroadcastShape) {
  for (const auto &test_case : test_cases) {
    BOOST_TEST(popart::npBroadcastable(test_case.a_shape, test_case.b_shape));

    const auto new_shape = popart::npOut(test_case.a_shape, test_case.b_shape);
    BOOST_TEST(new_shape == test_case.result_shape,
               boost::test_tools::per_element());
  }
}

BOOST_AUTO_TEST_CASE(NumpyBroadcastTensorInfo) {
  for (const auto &test_case : test_cases) {
    BOOST_TEST(popart::npBroadcastable(
        popart::TensorInfo(popart::DataType::FLOAT16, test_case.a_shape),
        popart::TensorInfo(popart::DataType::FLOAT16, test_case.b_shape)));

    const auto new_tensor = popart::npOut(
        popart::TensorInfo(popart::DataType::FLOAT16, test_case.a_shape),
        popart::TensorInfo(popart::DataType::FLOAT16, test_case.b_shape));
    BOOST_TEST(new_tensor.shape() == test_case.result_shape,
               boost::test_tools::per_element());
    BOOST_TEST(new_tensor.dataType() == popart::DataType::FLOAT16);
  }
}

BOOST_AUTO_TEST_CASE(NumpyBroadcastTensoInfoDataTypeMismatch) {
  for (const auto &test_case : test_cases) {
    BOOST_TEST(!popart::npBroadcastable(
        popart::TensorInfo(popart::DataType::FLOAT16, test_case.a_shape),
        popart::TensorInfo(popart::DataType::INT32, test_case.b_shape)));

    const size_t ERR_LEN = 500;

    const auto addShape = [&](const std::vector<int64_t> &shape,
                              std::ostream &os) {
      auto it = shape.begin();
      if (it == shape.end()) {
        return;
      }

      while (true) {
        os << *it;
        it++;
        if (it == shape.end()) {
          return;
        }
        os << " ";
      }
    };

    const auto predicate = [&](popart::error e) {
      std::ostringstream errm;
      errm << "np broadcasting failed, incompatible types FLOAT16 and INT32 ";
      errm << "(shapes [";
      addShape(test_case.a_shape, errm);
      errm << "] and [";
      addShape(test_case.b_shape, errm);
      errm << "])";

      return e.what() == errm.str();
    };

    BOOST_CHECK_EXCEPTION(
        popart::npOut(
            popart::TensorInfo(popart::DataType::FLOAT16, test_case.a_shape),
            popart::TensorInfo(popart::DataType::INT32, test_case.b_shape)),
        popart::error,
        predicate);
  }
}

struct BroadcastBackwardTestCase {
  std::vector<int64_t> in_shape;
  std::vector<int64_t> out_shape;
  std::vector<int64_t> result_axes;
};

// clang-format off
static const BroadcastBackwardTestCase backward_test_cases[] = {
    {{7, 2, 3, 4, 5, 6}, {7, 2, 3, 4, 5, 6}, {                }},
    {{2, 3, 4, 5, 6   }, {7, 2, 3, 4, 5, 6}, {0               }},
    {{3, 4, 5, 6      }, {7, 2, 3, 4, 5, 6}, {0, 1            }},
    {{4, 5, 6         }, {7, 2, 3, 4, 5, 6}, {0, 1, 2         }},
    {{5, 6            }, {7, 2, 3, 4, 5, 6}, {0, 1, 2, 3      }},
    {{6               }, {7, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4   }},
    {{                }, {7, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5}},
    {{1, 1, 1, 1, 1, 1}, {7, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5}},
    {{1, 1, 1, 1, 1, 6}, {7, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4   }},
    {{1, 1, 1, 1, 5, 6}, {7, 2, 3, 4, 5, 6}, {0, 1, 2, 3      }},
    {{1, 1, 1, 4, 5, 6}, {7, 2, 3, 4, 5, 6}, {0, 1, 2         }},
    {{1, 1, 3, 4, 5, 6}, {7, 2, 3, 4, 5, 6}, {0, 1            }},
    {{1, 2, 3, 4, 5, 6}, {7, 2, 3, 4, 5, 6}, {0               }},
    {{7, 2, 3, 4, 5, 6}, {7, 2, 3, 4, 5, 6}, {                }},

    // Test cases taken from
    // https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
    {{   256, 256, 3}, {   256, 256, 3}, {    }},
    {{8,   1,   6, 1}, {8,   7,   6, 5}, {1, 3}},
    {{             3}, {   256, 256, 3}, {0, 1}},
    {{     7,   1, 5}, {8,   7,   6, 5}, {0, 2}},
    {{             1}, {          5, 4}, {0, 1}},
    {{             4}, {          5, 4}, {   0}},
    {{    15,   1, 5}, {    15,   3, 5}, {   1}},
    {{          3, 5}, {    15,   3, 5}, {   0}},
    {{          3, 1}, {    15,   3, 5}, {0, 2}},

    // Test cases taken from
    // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    {{        4, 5}, {2, 3, 4, 5}, {      0, 1}},
    {{     1, 4, 5}, {2, 3, 4, 5}, {      0, 1}},
    {{     3, 4, 5}, {2, 3, 4, 5}, {         0}},
    {{/* scalar */}, {2, 3, 4, 5}, {0, 1, 2, 3}},
    {{           5}, {2, 3, 4, 5}, {   0, 1, 2}},
    {{  2, 3, 1, 1}, {2, 3, 4, 5}, {      2, 3}},
    {{  2, 1, 1, 1}, {2, 3, 4, 5}, {   1, 2, 3}}

};
// clang-format on

BOOST_AUTO_TEST_CASE(NumpyBroadcastBackwardShape) {
  for (const auto &test_case : backward_test_cases) {
    const auto axes =
        popart::npReductionAxis(test_case.in_shape, test_case.out_shape);
    BOOST_TEST(axes == test_case.result_axes, boost::test_tools::per_element());
  }
}

struct ExceptionTestCase {
  std::string name;
  std::vector<int64_t> a_shape;
  std::vector<int64_t> b_shape;
  std::string msg;
};

// clang-format off
ExceptionTestCase exception_test_cases[] = {
    {""   , {   3}, {   4}, "np broadcasting failed, frames [3] and [4] are not aligned"},
    {""   , {1, 3}, {   4}, "np broadcasting failed, frames [1, 3] and [4] are not aligned"},
    {""   , {4, 3}, {   4}, "np broadcasting failed, frames [4, 3] and [4] are not aligned"},
    {"foo", {   3}, {   4}, "np broadcasting failed on 'foo', frames [3] and [4] are not aligned"},
    {"foo", {1, 3}, {   4}, "np broadcasting failed on 'foo', frames [1, 3] and [4] are not aligned"},
    {"foo", {4, 3}, {   4}, "np broadcasting failed on 'foo', frames [4, 3] and [4] are not aligned"},
    {""   , {   3}, {3, 4}, "np broadcasting failed, frames [3] and [3, 4] are not aligned"},
    {""   , {   3}, {1, 4}, "np broadcasting failed, frames [3] and [1, 4] are not aligned"},
    {""   , {   3}, {   4}, "np broadcasting failed, frames [3] and [4] are not aligned"},
    {"foo", {   3}, {3, 4}, "np broadcasting failed on 'foo', frames [3] and [3, 4] are not aligned"},
    {"foo", {   3}, {1, 4}, "np broadcasting failed on 'foo', frames [3] and [1, 4] are not aligned"},
    {"foo", {   3}, {   4}, "np broadcasting failed on 'foo', frames [3] and [4] are not aligned"},
};
// clang-format on

BOOST_AUTO_TEST_CASE(NumpyBroadcastException) {
  for (const auto &test_case : exception_test_cases) {
    const auto predicate = [&](popart::error e) {
      return test_case.msg == e.what();
    };

    BOOST_TEST(!popart::npBroadcastable(test_case.a_shape, test_case.b_shape));

    BOOST_CHECK_EXCEPTION(
        popart::npOut(test_case.a_shape, test_case.b_shape, test_case.name),
        popart::error,
        predicate);
  }
}

BOOST_AUTO_TEST_CASE(NumpyBroadcastTensorInfoShapeException) {
  for (const auto &test_case : exception_test_cases) {

    // Skip tests with a debug name as not supported for TensorInfo use
    if (test_case.name != "") {
      continue;
    }

    const auto predicate = [&](popart::error e) {
      return test_case.msg == e.what();
    };

    BOOST_TEST(!popart::npBroadcastable(
        popart::TensorInfo(popart::DataType::FLOAT16, test_case.a_shape),
        popart::TensorInfo(popart::DataType::FLOAT16, test_case.b_shape)));

    BOOST_CHECK_EXCEPTION(
        popart::npOut(
            popart::TensorInfo(popart::DataType::FLOAT16, test_case.a_shape),
            popart::TensorInfo(popart::DataType::FLOAT16, test_case.b_shape)),
        popart::error,
        predicate);
  }
}