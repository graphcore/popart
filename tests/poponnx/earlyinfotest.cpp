#define BOOST_TEST_MODULE EarlyUnfoTest

#include <boost/test/unit_test.hpp>

#include <poponnx/earlyinfo.hpp>
#include <poponnx/error.hpp>
#include <poponnx/tensorinfo.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(EarlyInfo_Case1) {
  auto ei = poponnx::EarlyInfo();

  BOOST_CHECK(ei.has("cat") == false);
  BOOST_CHECK(ei.getAllTensorIds().size() == 0);
  BOOST_CHECK_THROW(ei.get("cat"), poponnx::error);

  poponnx::TensorInfo input("FLOAT", std::vector<int64_t>({2, 2}));
  ei.add("cat", input);

  BOOST_CHECK(ei.has("cat") == true);
  BOOST_CHECK(ei.getAllTensorIds().size() == 1);
  auto &output = ei.get("cat");

  BOOST_CHECK(input == output);

  poponnx::EarlyInfo ei2(ei);

  BOOST_CHECK(ei.has("cat") == true);
  BOOST_CHECK(ei2.has("cat") == true);
}
