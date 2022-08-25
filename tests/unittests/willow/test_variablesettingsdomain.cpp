// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_variablesettings

#include <vector>

#include "boost/test/data/test_case.hpp"
#include "boost/test/unit_test.hpp"
#include "popart/commgroup.hpp"
#include "popart/replicagrouping.hpp"
#include "variablesettingsdomain.hpp"

struct TestVariablesSettingsDomainConstructorData {
  const popart::VariableSettingsDomain domain;
  bool hasCommGroup;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestVariablesSettingsDomainConstructorData)

std::vector<TestVariablesSettingsDomainConstructorData>
    testVariablesSettingsDomainConstructorData{
        {popart::ReplicaGrouping(1), false},
        {{1, 1, 1}, false},
        {popart::CommGroup(), true},
        {{popart::CommGroupType::All, 0}, true},
    };

BOOST_DATA_TEST_CASE(
    testVariablesSettingsDomainConstructor,
    ::boost::unit_test::data::make(testVariablesSettingsDomainConstructorData),
    testData) {
  const auto &domain       = testData.domain;
  const auto &hasCommGroup = testData.hasCommGroup;

  BOOST_CHECK_EQUAL(domain.commGroup_.has_value(), hasCommGroup);
  BOOST_CHECK_EQUAL(domain.grouping_.has_value(), !hasCommGroup);
}

struct TestVariablesSettingsDomainData {
  const popart::VariableSettingsDomain lhs;
  const popart::VariableSettingsDomain rhs;
  const bool expectedResult;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestVariablesSettingsDomainData)

const std::vector<TestVariablesSettingsDomainData>
    testVariablesSettingsDomainData{
        {popart::ReplicaGrouping(1), popart::ReplicaGrouping(2), false},
        {popart::CommGroup(popart::CommGroupType::All, 0),
         popart::CommGroup(popart::CommGroupType::None, 0),
         false},
        {popart::ReplicaGrouping(1), popart::CommGroup(), false},
        {popart::ReplicaGrouping(1), popart::ReplicaGrouping(1), true},
        {popart::CommGroup(), popart::CommGroup(), true},
    };

BOOST_DATA_TEST_CASE(
    testVariablesSettingsDomain,
    ::boost::unit_test::data::make(testVariablesSettingsDomainData),
    testData) {
  const auto &lhs            = testData.lhs;
  const auto &rhs            = testData.rhs;
  const auto &expectedResult = testData.expectedResult;

  BOOST_CHECK_EQUAL(lhs == rhs, expectedResult);
  BOOST_CHECK_EQUAL(lhs != rhs, !expectedResult);
}
