#define BOOST_TEST_MODULE NumpyBroadcastShape

#include <boost/test/unit_test.hpp>
#include <poponnx/error.hpp>
#include <poponnx/tensorinfo.hpp>

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
    const auto new_shape = poponnx::npOut(test_case.a_shape, test_case.b_shape);
    BOOST_TEST(new_shape == test_case.result_shape,
               boost::test_tools::per_element());
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
        poponnx::npReductionAxis(test_case.in_shape, test_case.out_shape);
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
    const auto predicate = [&](poponnx::error e) {
      return test_case.msg == e.what();
    };

    BOOST_CHECK_EXCEPTION(
        poponnx::npOut(test_case.a_shape, test_case.b_shape, test_case.name),
        poponnx::error,
        predicate);
  }
}
