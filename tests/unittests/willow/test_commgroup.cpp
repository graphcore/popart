// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_commgroup

#include <vector>

#include "boost/test/data/test_case.hpp"
#include "boost/test/unit_test.hpp"
#include "popart/commgroup.hpp"
#include "popart/error.hpp"
#include "popart/replicagrouping.hpp"

struct TestReplicaGroupingCommGroupConversionData {
  const popart::CommGroup commGroup;
  const popart::ReplicaGrouping replicaGrouping;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestReplicaGroupingCommGroupConversionData)

const std::vector<TestReplicaGroupingCommGroupConversionData>
    testReplicaGroupingToCommGroupData{
        {{popart::CommGroupType::All, 0}, {4, 1, 4}},
        {{popart::CommGroupType::Consecutive, 2}, {4, 1, 2}},
        {{popart::CommGroupType::Orthogonal, 2}, {4, 2, 2}},
        {{popart::CommGroupType::None, 0}, {4, 2, 1}},
    };

BOOST_DATA_TEST_CASE(
    testReplicaGroupingToCommGroup,
    ::boost::unit_test::data::make(testReplicaGroupingToCommGroupData),
    testData) {
  const auto &commGroup       = testData.commGroup;
  const auto &replicaGrouping = testData.replicaGrouping;

  BOOST_CHECK_EQUAL(popart::CommGroup(replicaGrouping), commGroup);
}

BOOST_AUTO_TEST_CASE(testReplicaGroupingToCommGroupInvalid) {
  BOOST_CHECK_THROW(
      popart::CommGroup(popart::CommGroup(popart::ReplicaGrouping(8, 2, 2))),
      popart::error);
}

const std::vector<TestReplicaGroupingCommGroupConversionData>
    testToCommGroupReplicaGroupingData{
        {{popart::CommGroupType::All, 0}, {4, 1, 4}},
        {{popart::CommGroupType::Consecutive, 2}, {4, 1, 2}},
        {{popart::CommGroupType::Orthogonal, 2}, {4, 2, 2}},
        {{popart::CommGroupType::None, 0}, {4, 2, 1}},
    };

BOOST_DATA_TEST_CASE(
    testToCommGroupReplicaGrouping,
    ::boost::unit_test::data::make(testToCommGroupReplicaGroupingData),
    testData) {
  const auto &commGroup       = testData.commGroup;
  const auto &replicaGrouping = testData.replicaGrouping;
  const auto &numReplicas     = replicaGrouping.getNumReplicas();

  BOOST_CHECK_EQUAL(commGroup.toReplicaGrouping(numReplicas), replicaGrouping);
}

struct TestEqualityOperatorData {
  const popart::CommGroup lhs;
  const popart::CommGroup rhs;
  const bool expectedResult;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestEqualityOperatorData)

const std::vector<TestEqualityOperatorData> testEqualityOperatorData{
    {{popart::CommGroupType::All, 0}, {popart::CommGroupType::All, 0}, true},
    {{popart::CommGroupType::All, 0}, {popart::CommGroupType::None, 0}, false},
    {{popart::CommGroupType::All, 0}, {popart::CommGroupType::All, 1}, false},
};

BOOST_DATA_TEST_CASE(testEqualityOperator,
                     ::boost::unit_test::data::make(testEqualityOperatorData),
                     testData) {
  const auto &lhs            = testData.lhs;
  const auto &rhs            = testData.rhs;
  const auto &expectedResult = testData.expectedResult;

  BOOST_CHECK_EQUAL(lhs == rhs, expectedResult);
  BOOST_CHECK_EQUAL(lhs != rhs, !expectedResult);
}
