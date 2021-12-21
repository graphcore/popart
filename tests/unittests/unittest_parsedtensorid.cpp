// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ParsedTensorIdTest

#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/scope.hpp>
#include <popart/tensornames.hpp>

#include <parsedtensorid.hpp>

struct ParsedTensorIdFixture {
  // Fixture available in all fixtured tests
  ParsedTensorIdFixture()
      : d(popart::sNameDelimiter), p1(popart::reservedGradientPrefix()),
        p2(popart::reservedUpdatedVarPrefix()),
        p3(popart::reservedAccumPrefix()), s1("scope1"), s2("scope2"),
        s3("scope3") {
    // Create graphs so that the scopes will be parsed
    ir.createGraph({s1});
    ir.createGraph({s2});
    ir.createGraph({s3});
    ir.createGraph({"g1"});
    ir.createGraph({"g2"});
    ir.createGraph({"g3"});
    // Create graphs so that the scopes will be parsed
    scope1 = scope1 / s1;
    scope2 = scope2 / s2 / s3;
  }

  bool checkErrorMsgAddScopes(const popart::error &ex) {
    const auto expectedPrefix = "Cannot add scope ";
    return boost::algorithm::starts_with(ex.what(), expectedPrefix);
  }

  bool checkErrorMsgRemoveScopes(const popart::error &ex) {
    const auto expectedPrefix = "Cannot remove scope ";
    return boost::algorithm::starts_with(ex.what(), expectedPrefix);
  }

  popart::Ir ir;

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
  // Tests ParsedTensorId for the most basic cases
  std::string name     = "basic";
  popart::TensorId tId = name;
  popart::ParsedTensorId pTId(tId, ir);
  BOOST_CHECK_EQUAL(pTId.getId(), name);

  name = "";
  tId  = name;
  pTId = {tId, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), name);
}

BOOST_AUTO_TEST_CASE(testParsingParsedTensorId) {
  // Tests that the parser can handle parsing of scopes and prefixes

  // Test parsing when prefix is not added by the function
  std::string name     = std::string(p1) + "testTIdScope";
  popart::TensorId tId = name;
  popart::ParsedTensorId pTId(tId, ir);
  BOOST_CHECK_EQUAL(pTId.getId(), name);
  BOOST_ASSERT(pTId.prefixExist(p1));

  // Test parsing when scope is not added by the function
  popart::Scope g1;
  g1   = g1 / "g1";
  name = "g1/testTIdScope";
  pTId = {name, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), name);
  BOOST_ASSERT(pTId.scopeExist(g1));

  // Test parsing when the scopes starts with the same name
  // I.e. g(1/
  popart::Scope g1g11g111;
  g1g11g111 = g1g11g111 / "g(1/1)" / "g(1/)" / "g(1/11)";
  ir.createGraph({"g(1/1)"});
  ir.createGraph({"g(1/)"});
  ir.createGraph({"g(1/11)"});
  name = "g(1/1)/g(1/)/g(1/11)/testTIdScope";
  pTId = {name, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), name);
  BOOST_ASSERT(pTId.scopeExist(g1g11g111));

  // Test when a scope is fully contained in another
  popart::Scope g;
  g = g / "g";
  ir.createGraph({"g"});
  name = "g1/g/myName";
  pTId = {name, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), "g1/g/myName");
  BOOST_ASSERT(pTId.scopeExist(g1));
  BOOST_ASSERT(pTId.scopeExist(g));

  // Test limitation: The scope is not in the start
  name = "foo_g1_bar";
  pTId = {name, ir};
  // The following check shows the limitation
  BOOST_CHECK_EQUAL(pTId.getId(), "g1/bar");
}

BOOST_AUTO_TEST_CASE(testParsedTensorIdPrefixes) {
  // Tests adding and removing of prefixes when the TensorId is without scopes
  std::string name = "testParsedTensorIdPrefixes";
  popart::TensorId tId(name);
  popart::ParsedTensorId pTId(tId, ir);
  BOOST_CHECK_EQUAL(pTId.getId(), name);
  BOOST_ASSERT(!pTId.prefixExist(p1));
  BOOST_ASSERT(!pTId.prefixExist(p2));
  BOOST_ASSERT(!pTId.prefixExist(p3));

  // Add prefix
  pTId.addPrefix(p1);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + name);
  BOOST_ASSERT(pTId.prefixExist(p1));
  BOOST_ASSERT(!pTId.prefixExist(p2));
  BOOST_ASSERT(!pTId.prefixExist(p3));

  pTId.addPrefix(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + p2 + name);
  BOOST_ASSERT(pTId.prefixExist(p1));
  BOOST_ASSERT(pTId.prefixExist(p2));
  BOOST_ASSERT(!pTId.prefixExist(p3));

  pTId.addPrefix(p3);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + p2 + p3 + name);
  BOOST_ASSERT(pTId.prefixExist(p1));
  BOOST_ASSERT(pTId.prefixExist(p2));
  BOOST_ASSERT(pTId.prefixExist(p3));

  // Remove prefix
  pTId.removePrefixIfExist(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + p3 + name);
  BOOST_ASSERT(pTId.prefixExist(p1));
  BOOST_ASSERT(!pTId.prefixExist(p2));
  BOOST_ASSERT(pTId.prefixExist(p3));

  pTId.removePrefixIfExist(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), p1 + p3 + name);
  BOOST_ASSERT(pTId.prefixExist(p1));
  BOOST_ASSERT(!pTId.prefixExist(p2));
  BOOST_ASSERT(pTId.prefixExist(p3));
}

BOOST_AUTO_TEST_CASE(testParsedTensorIdScopes) {
  // Tests adding and removing of scopes when the TensorId is without prefixes
  std::string name = "testTIdScope";
  popart::TensorId tId(name);
  popart::ParsedTensorId pTId(tId, ir);
  BOOST_CHECK_EQUAL(pTId.getId(), name);
  BOOST_ASSERT(!pTId.scopeExist(scope1));
  BOOST_ASSERT(!pTId.scopeExist(scope2));

  // Add Scope
  pTId.addScope(scope1);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + name);
  BOOST_ASSERT(pTId.scopeExist(scope1));
  BOOST_ASSERT(!pTId.scopeExist(scope2));

  pTId.addScope(scope2);
  BOOST_CHECK_EQUAL(pTId.getId(), s2 + d + s3 + d + s1 + d + name);
  BOOST_ASSERT(pTId.scopeExist(scope1));
  BOOST_ASSERT(pTId.scopeExist(scope2));

  pTId.removeScope(scope2);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + name);
  BOOST_ASSERT(pTId.scopeExist(scope1));
  BOOST_ASSERT(!pTId.scopeExist(scope2));

  BOOST_CHECK_EXCEPTION(
      pTId.removeScope(scope2), popart::error, checkErrorMsgRemoveScopes);

  // Test with empty scope
  popart::Scope emptyScope;
  name = "empty";
  pTId = {name, ir};
  pTId.addScope(emptyScope);
  BOOST_CHECK_EQUAL(pTId.getId(), name);
  pTId.removeScope(emptyScope);
  BOOST_CHECK_EQUAL(pTId.getId(), name);
  BOOST_ASSERT(!pTId.scopeExist(scope1));
  BOOST_ASSERT(!pTId.scopeExist(scope2));

  // Test with scope not present in the IR
  std::string tIdStr = "testTIdScope";
  popart::Scope scopeNotInIr;
  std::string scopeNotInIrStr = "scopeNotInIr";
  scopeNotInIr                = scopeNotInIr / scopeNotInIrStr;
  pTId                        = {tIdStr, ir};
  BOOST_CHECK_EXCEPTION(
      pTId.addScope(scopeNotInIr), popart::error, checkErrorMsgAddScopes);

  name = scopeNotInIrStr + d + scopeNotInIrStr;
  pTId = {name, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), name);
  BOOST_CHECK_EXCEPTION(
      pTId.removeScope(scopeNotInIr), popart::error, checkErrorMsgRemoveScopes);
  BOOST_ASSERT(!pTId.scopeExist(scopeNotInIr));
}

BOOST_AUTO_TEST_CASE(testParsedTensorIdMixedScopePrefixesAndNames) {
  // Tests adding and removing of scopes and prefixes without restrictions on
  // TensorId
  std::string name = "testParsedTensorIdMixedScopePrefixesAndNames";
  popart::TensorId tId(name);
  popart::ParsedTensorId pTId(tId, ir);
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

  // TensorId containing / in the name
  popart::Scope scopeNotInIr;
  scopeNotInIr = scopeNotInIr / "MatMul:0";
  name         = scopeNotInIr.str() + d + "5";
  tId          = s1 + d + name;
  pTId         = {tId, ir};
  BOOST_ASSERT(!pTId.scopeExist(scopeNotInIr));
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + name);
  pTId.addPrefix(p1);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + p1 + name);
  pTId.addPrefix(p2);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + p1 + p2 + name);
  pTId.addScope(scope2);
  BOOST_CHECK_EQUAL(pTId.getId(), s2 + d + s3 + d + s1 + d + p1 + p2 + name);
  pTId.removePrefixIfExist(p1);
  BOOST_CHECK_EQUAL(pTId.getId(), s2 + d + s3 + d + s1 + d + p2 + name);
  pTId.removeScope(scope2);
  BOOST_CHECK_EQUAL(pTId.getId(), s1 + d + p2 + name);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_CASE(testDocumentationParsedTensorId) {
  // Test of documentation
  popart::Ir ir;
  ir.createGraph({"g1"});
  ir.createGraph({"g2"});
  ir.createGraph({"g3"});

  popart::TensorId tId = "g1/g2/Step___Accl___name";
  popart::ParsedTensorId pTId(tId, ir);

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

BOOST_AUTO_TEST_CASE(testPruneOverlappedMatches) {
  // Test that the pruneOverlappedMatches function work as expected
  // Define the expected map
  std::map<std::size_t, std::size_t> expected;
  expected[0]  = 2;
  expected[3]  = 6;
  expected[11] = 2;
  expected[14] = 2;
  expected[21] = 5;

  // Add overlaps for strBeginAndStrLengths
  std::map<std::size_t, std::size_t> strBeginAndStrLengths = expected;
  strBeginAndStrLengths[4]                                 = 4;
  strBeginAndStrLengths[5] = 1; // Nested overlap
  // Overlap matching beginning is not possible as we are dealing with maps
  strBeginAndStrLengths[15] = 1; // Overlap matching end

  popart::pruneOverlappedMatches(strBeginAndStrLengths);

  BOOST_ASSERT(expected.size() == strBeginAndStrLengths.size());
  for (const auto valueKey : expected) {
    BOOST_CHECK_EQUAL(expected.at(valueKey.first),
                      strBeginAndStrLengths.at(valueKey.first));
  }
}

BOOST_AUTO_TEST_CASE(testPrefixOverlap) {
  // Test that there are no spurious hits when a prefix overlaps with another
  popart::Ir ir;

  popart::TensorId tId = std::string(popart::reservedConcatInitPrefix()) + "t1";
  popart::ParsedTensorId pTId(tId, ir);
  BOOST_CHECK_EQUAL(pTId.getId(), tId);

  tId = std::string(popart::reservedInitPrefix()) +
        std::string(popart::reservedConcatInitPrefix()) + "t1";
  pTId = {tId, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), tId);

  tId = std::string(popart::reservedConcatInitPrefix()) +
        std::string(popart::reservedInitPrefix()) + "t1";
  pTId = {tId, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), tId);

  tId = std::string(popart::reservedInitPrefix()) +
        std::string(popart::reservedConcatInitPrefix()) +
        std::string(popart::reservedInitPrefix()) + "t1";
  pTId = {tId, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), tId);
}

BOOST_AUTO_TEST_CASE(testScopeOverlap) {
  // Test that there are no spurious hits when a scope overlaps with another
  std::string g1  = "g1";
  std::string g12 = "g12";
  popart::Ir ir;
  ir.createGraph({g1});
  ir.createGraph({g12});

  popart::TensorId tId = g1 + "/" + g12 + "/";
  popart::ParsedTensorId pTId(tId, ir);
  BOOST_CHECK_EQUAL(pTId.getId(), tId);

  tId  = g12 + "/" + g1 + "/";
  pTId = {tId, ir};
  BOOST_CHECK_EQUAL(pTId.getId(), tId);
}
