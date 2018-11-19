
#define BOOST_TEST_MODULE DataFlowTest

#include <boost/test/unit_test.hpp>

#include <poponnx/ir.hpp>

using namespace willow;

BOOST_AUTO_TEST_CASE(DataFlow_Case1) {
  auto df = willow::DataFlow();

  BOOST_CHECK(df.nAnchors() == 0);
  BOOST_CHECK(df.batchSize() == 0);
  BOOST_CHECK(df.batchesPerStep() == 0);
  BOOST_CHECK(df.art() == AnchorReturnType::FINAL);
  BOOST_CHECK(df.isAnchored("one") == false);
}

BOOST_AUTO_TEST_CASE(DataFlow_Case2) {

  auto df = willow::DataFlow(5, 2, {"one", "two"}, AnchorReturnType::ALL);

  BOOST_CHECK(df.nAnchors() == 2);
  BOOST_CHECK(df.batchSize() == 2);
  BOOST_CHECK(df.batchesPerStep() == 5);
  BOOST_CHECK(df.art() == AnchorReturnType::ALL);
  BOOST_CHECK(df.isAnchored("two") == true);
  BOOST_CHECK(df.isAnchored("three") == false);

  willow::DataFlow df2(df);

  BOOST_CHECK(df2.nAnchors() == 2);
  BOOST_CHECK(df2.batchSize() == 2);
  BOOST_CHECK(df2.batchesPerStep() == 5);
  BOOST_CHECK(df2.art() == AnchorReturnType::ALL);
  BOOST_CHECK(df2.isAnchored("two") == true);
  BOOST_CHECK(df2.isAnchored("three") == false);
  BOOST_CHECK(df2.anchors()[0] == "one");

  willow::DataFlow df3 = df;

  BOOST_CHECK(df3.nAnchors() == 2);
  BOOST_CHECK(df3.batchSize() == 2);
  BOOST_CHECK(df3.batchesPerStep() == 5);
  BOOST_CHECK(df3.art() == AnchorReturnType::ALL);
  BOOST_CHECK(df3.isAnchored("two") == true);
  BOOST_CHECK(df3.isAnchored("three") == false);
  BOOST_CHECK(df3.anchors()[0] == "one");
}