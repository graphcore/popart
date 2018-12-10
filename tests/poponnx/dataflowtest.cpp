
#define BOOST_TEST_MODULE DataFlowTest

#include <boost/test/unit_test.hpp>

#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>

using namespace poponnx;

bool inValidReturnPeriod1(const error &ex) {
  BOOST_CHECK_EQUAL(ex.what(),
                    std::string("A return period should not be supplied for "
                                "this anchor return type"));
  return true;
}

bool inValidReturnPeriod2(const error &ex) {
  BOOST_CHECK_EQUAL(
      ex.what(),
      std::string(
          "Return period must be <= to the number of batches per step"));
  return true;
}

bool inValidReturnPeriod3(const error &ex) {
  BOOST_CHECK_EQUAL(ex.what(),
                    std::string("Return period must be a factor of the "
                                "number of batches per step"));
  return true;
}

BOOST_AUTO_TEST_CASE(DataFlow_Case1) {
  auto df = poponnx::DataFlow();

  BOOST_CHECK(df.nAnchors() == 0);
  BOOST_CHECK(df.batchSize() == 0);
  BOOST_CHECK(df.batchesPerStep() == 0);
  BOOST_CHECK(df.isAnchored("one") == false);
}

BOOST_AUTO_TEST_CASE(DataFlow_Case2) {

  auto df = poponnx::DataFlow(
      5,
      2,
      {{"one", AnchorReturnType("ALL")}, {"two", AnchorReturnType("FINAL")}});

  BOOST_CHECK(df.nAnchors() == 2);
  BOOST_CHECK(df.batchSize() == 2);
  BOOST_CHECK(df.batchesPerStep() == 5);
  BOOST_CHECK(df.art("one").id() == AnchorReturnTypeId::ALL);
  BOOST_CHECK(df.art("two").id() == AnchorReturnTypeId::FINAL);
  BOOST_CHECK(df.isAnchored("two") == true);
  BOOST_CHECK(df.isAnchored("three") == false);

  poponnx::DataFlow df2(df);

  BOOST_CHECK(df2.nAnchors() == 2);
  BOOST_CHECK(df2.batchSize() == 2);
  BOOST_CHECK(df2.batchesPerStep() == 5);
  BOOST_CHECK(df.art("one").id() == AnchorReturnTypeId::ALL);
  BOOST_CHECK(df.art("two").id() == AnchorReturnTypeId::FINAL);
  BOOST_CHECK(df2.isAnchored("two") == true);
  BOOST_CHECK(df2.isAnchored("three") == false);
  BOOST_CHECK(df2.anchors()[0] == "one");

  poponnx::DataFlow df3 = df;

  BOOST_CHECK(df3.nAnchors() == 2);
  BOOST_CHECK(df3.batchSize() == 2);
  BOOST_CHECK(df3.batchesPerStep() == 5);
  BOOST_CHECK(df.art("one").id() == AnchorReturnTypeId::ALL);
  BOOST_CHECK(df.art("two").id() == AnchorReturnTypeId::FINAL);
  BOOST_CHECK(df3.isAnchored("two") == true);
  BOOST_CHECK(df3.isAnchored("three") == false);
  BOOST_CHECK(df3.anchors()[0] == "one");
}

BOOST_AUTO_TEST_CASE(DataFlow_Case3) {

  auto df = poponnx::DataFlow(6,
                              2,
                              {{"one", AnchorReturnType("EVERYN", 2)},
                               {"two", AnchorReturnType("FINAL")}});

  BOOST_CHECK(df.art("one").rp() == 2);
  BOOST_CHECK_EXCEPTION(df.art("two").rp(), error, inValidReturnPeriod1);
}

BOOST_AUTO_TEST_CASE(DataFlow_Case4) {

  auto art = AnchorReturnType("EVERYN", 6);
  BOOST_CHECK_EXCEPTION(
      poponnx::DataFlow(5, 2, {{"one", art}}), error, inValidReturnPeriod2);
}

BOOST_AUTO_TEST_CASE(DataFlow_Case5) {

  auto art = AnchorReturnType("EVERYN", 3);
  BOOST_CHECK_EXCEPTION(
      poponnx::DataFlow(5, 2, {{"one", art}}), error, inValidReturnPeriod3);
}
