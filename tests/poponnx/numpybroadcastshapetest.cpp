#define BOOST_TEST_MODULE NumpyBroadcastShape

#include <boost/test/unit_test.hpp>
#include <poponnx/tensorinfo.hpp>

struct TestCase {
  std::vector<int64_t> a_shape;
  std::vector<int64_t> b_shape;
  std::vector<int64_t> result_shape;
};

// clang-format off
static const TestCase test_cases[] = {
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
    const auto new_shape = willow::npOut(test_case.a_shape, test_case.b_shape);
    BOOST_TEST(new_shape == test_case.result_shape,
               boost::test_tools::per_element());
  }
}
