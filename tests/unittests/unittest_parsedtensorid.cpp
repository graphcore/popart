// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ParsedTensorIdTest

#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/scope.hpp>
#include <popart/tensornames.hpp>

#include <parsedtensorid.hpp>

struct ParsedTensorIdFixture {
  ParsedTensorIdFixture()
      : d(popart::sNameDelimiter), p1(popart::reservedGradientPrefix()),
        p2(popart::reservedUpdatedVarPrefix()),
        p3(popart::reservedAccumPrefix()), s1("scope1"), s2("scope2"),
        s3("scope3") {
    scope1 = scope1 / s1;
    scope2 = scope2 / s2 / s3;
  }

  bool checkErrorMsgRemoveScopes(const popart::error &ex) {
    const auto expectedPrefix = "Cannot remove scope from ";
    return boost::algorithm::starts_with(ex.what(), expectedPrefix);
  }

  std::string d;

  const char *p1;
  std::string p2;
  std::string p3;

  std::string s1;
  std::string s2;
  std::string s3;

  popart::Scope scope1;
  popart::Scope scope2;
};

BOOST_FIXTURE_TEST_SUITE(ParsedTensorIdTestSuite, ParsedTensorIdFixture)

BOOST_AUTO_TEST_CASE(testBasicParsedTensorId) {
  std::string testStr  = "basic";
  popart::TensorId tId = testStr;
  popart::ParsedTensorId pTId(tId);
  BOOST_CHECK_EQUAL(pTId.getId(), testStr);

  testStr = "";
  tId     = testStr;
  pTId    = tId;
  BOOST_CHECK_EQUAL(pTId.getId(), testStr);
}

BOOST_AUTO_TEST_CASE(testParsedTensorIdPrefixes) {
  std::string name = "testParsedTensorIdPrefixes";
  popart::TensorId tId(name);
  popart::ParsedTensorId pTId(tId);
  BOOST_CHECK_EQUAL(pTId.getId(), name);

  // Add prefix
  pTId.addPrefix(p1);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + name);

  pTId.addPrefix(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + p2 + name);

  pTId.addPrefix(p3);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + p2 + p3 + name);

  // Remove prefix
  pTId.removePrefixIfExist(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + p3 + name);

  pTId.removePrefixIfExist(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + p3 + name);
}

BOOST_AUTO_TEST_CASE(testParsedTensorIdScopes) {
  std::string name = "testTIdScope";
  popart::TensorId tId(name);
  popart::ParsedTensorId pTId(tId);
  BOOST_CHECK_EQUAL(pTId.getId(), name);

  // Add Scope
  pTId.addScope(scope1);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + name);

  pTId.addScope(scope2);
  BOOST_CHECK_EQUAL(pTId.getId(), s2 + d + s3 + d + s1 + d + name);

  pTId.removeScope(scope2);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + name);

  BOOST_CHECK_EXCEPTION(
      pTId.removeScope(scope2), popart::error, checkErrorMsgRemoveScopes);
}

BOOST_AUTO_TEST_CASE(testParsedTensorIdMixedScopePrefixesAndNames) {
  std::string name = "testParsedTensorIdMixedScopePrefixesAndNames";
  popart::TensorId tId(name);
  popart::ParsedTensorId pTId(tId);
  BOOST_CHECK_EQUAL(pTId.getId(), name);

  pTId.addPrefix(p1);
  pTId.addScope(scope1);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + p1 + name);

  // Add scope when there is a prefix
  pTId.addScope(scope2);
  BOOST_CHECK_EQUAL(pTId.getId(), s2 + d + s3 + d + s1 + d + p1 + name);

  // Add the rest of the prefixes
  pTId.addPrefix(p2);
  pTId.addPrefix(p3);
  BOOST_CHECK_EQUAL(pTId.getId(),
                    s2 + d + s3 + d + s1 + d + p1 + p2 + p3 + name);

  // Remove a prefix in the middle
  pTId.removePrefixIfExist(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), s2 + d + s3 + d + s1 + d + p1 + p3 + name);
  pTId.removePrefixIfExist(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), s2 + d + s3 + d + s1 + d + p1 + p3 + name);

  // Remove the scope
  pTId.removeScope(scope2);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + p1 + p3 + name);

  BOOST_CHECK_EXCEPTION(
      pTId.removeScope(scope2), popart::error, checkErrorMsgRemoveScopes);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_CASE(testDocumentationParsedTensorId) {
  popart::TensorId tId = "g1/g2/Step___Accl___name";
  popart::ParsedTensorId pTId(tId);

  popart::Scope graphScope1;
  graphScope1 = graphScope1 / "g1";
  popart::Scope newGraphScope;
  newGraphScope = newGraphScope / "g3";
  pTId.removeScope(graphScope1);
  pTId.addScope(newGraphScope);

  pTId.removePrefixIfExist(popart::reservedAcclPrefix());
  pTId.addPrefix(popart::reservedGradientPrefix());

  popart::TensorId expected = "g3/g2/Step___Gradient___name";
  BOOST_CHECK_EQUAL(pTId.getId(), expected);
}