
#define BOOST_TEST_MODULE EarlyUnfoTest

#include <boost/test/unit_test.hpp>

#include <poponnx/ir.hpp>

using namespace willow;

BOOST_AUTO_TEST_CASE(EarlyInfo_Case1) {
  auto ei = willow::EarlyInfo();

  BOOST_CHECK(ei.has("cat") == false);
  BOOST_CHECK(ei.getAllTensorIds().size() == 0);
  BOOST_CHECK_THROW(ei.get("cat"), willow::error);

  willow::TensorInfo input("FLOAT", std::vector<int64_t>({2, 2}));
  ei.add("cat", input);

  BOOST_CHECK(ei.has("cat") == true);
  BOOST_CHECK(ei.getAllTensorIds().size() == 1);
  auto &output = ei.get("cat");

  BOOST_CHECK(input == output);

  willow::EarlyInfo ei2(ei);

  BOOST_CHECK(ei.has("cat") == true);
  BOOST_CHECK(ei2.has("cat") == true);
}
