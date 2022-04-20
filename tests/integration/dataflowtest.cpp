// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE DataFlowTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>
#include <popart/error.hpp>

#include "popart/dataflow.hpp"
#include "popart/erroruid.hpp"
#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"

using namespace popart;

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
  auto df = popart::DataFlow();

  BOOST_CHECK(df.nAnchors() == 0);
  BOOST_CHECK(df.batchesPerStep() == 0);
  BOOST_CHECK(df.isAnchored("one") == false);
}

BOOST_AUTO_TEST_CASE(DataFlow_Case2) {

  auto df = popart::DataFlow(
      5,
      {{"one", AnchorReturnType("All")}, {"two", AnchorReturnType("Final")}});

  BOOST_CHECK(df.nAnchors() == 2);
  BOOST_CHECK(df.batchesPerStep() == 5);
  BOOST_CHECK(df.art("one").id() == AnchorReturnTypeId::All);
  BOOST_CHECK(df.art("two").id() == AnchorReturnTypeId::Final);
  BOOST_CHECK(df.isAnchored("two") == true);
  BOOST_CHECK(df.isAnchored("three") == false);

  popart::DataFlow df2(df);

  BOOST_CHECK(df2.nAnchors() == 2);
  BOOST_CHECK(df2.batchesPerStep() == 5);
  BOOST_CHECK(df.art("one").id() == AnchorReturnTypeId::All);
  BOOST_CHECK(df.art("two").id() == AnchorReturnTypeId::Final);
  BOOST_CHECK(df2.isAnchored("two") == true);
  BOOST_CHECK(df2.isAnchored("three") == false);
  BOOST_CHECK(df2.anchors()[0] == "one");

  popart::DataFlow df3 = df;

  BOOST_CHECK(df3.nAnchors() == 2);
  BOOST_CHECK(df3.batchesPerStep() == 5);
  BOOST_CHECK(df.art("one").id() == AnchorReturnTypeId::All);
  BOOST_CHECK(df.art("two").id() == AnchorReturnTypeId::Final);
  BOOST_CHECK(df3.isAnchored("two") == true);
  BOOST_CHECK(df3.isAnchored("three") == false);
  BOOST_CHECK(df3.anchors()[0] == "one");
}

BOOST_AUTO_TEST_CASE(DataFlow_Case3) {

  auto df = popart::DataFlow(6,
                             {{"one", AnchorReturnType("EveryN", 2)},
                              {"two", AnchorReturnType("Final")}});

  BOOST_CHECK(df.art("one").rp() == 2);
  BOOST_CHECK_EXCEPTION(df.art("two").rp(), error, inValidReturnPeriod1);
}

BOOST_AUTO_TEST_CASE(DataFlow_Case4) {

  auto art = AnchorReturnType("EveryN", 6);
  BOOST_CHECK_EXCEPTION(
      popart::DataFlow(5, {{"one", art}}), error, inValidReturnPeriod2);
}

BOOST_AUTO_TEST_CASE(DataFlow_Case5) {

  auto art = AnchorReturnType("EveryN", 3);
  BOOST_CHECK_EXCEPTION(
      popart::DataFlow(5, {{"one", art}}), error, inValidReturnPeriod3);
}

BOOST_AUTO_TEST_CASE(DataFlow_Case6) {
  std::vector<TensorId> ids{"one", "two"};

  auto df = popart::DataFlow(3, ids);
  BOOST_CHECK(df.nAnchors() == 2);
  BOOST_CHECK(df.batchesPerStep() == 3);
  BOOST_CHECK(df.isAnchored("one") == true);
  BOOST_CHECK(df.isAnchored("two") == true);
  BOOST_CHECK(df.art("one").id() == AnchorReturnTypeId::All);
  BOOST_CHECK(df.art("two").id() == AnchorReturnTypeId::All);

  auto df1 = popart::DataFlow(3, ids, AnchorReturnType("Final"));
  BOOST_CHECK(df1.nAnchors() == 2);
  BOOST_CHECK(df1.batchesPerStep() == 3);
  BOOST_CHECK(df1.isAnchored("one") == true);
  BOOST_CHECK(df1.isAnchored("two") == true);
  BOOST_CHECK(df1.art("one").id() == AnchorReturnTypeId::Final);
  BOOST_CHECK(df1.art("two").id() == AnchorReturnTypeId::Final);
}
