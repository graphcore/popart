// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_variablesettings

#include <vector>

#include "boost/test/data/test_case.hpp"
#include "boost/test/unit_test.hpp"
#include "popart/commgroup.hpp"
#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/replicagrouping.hpp"
#include "popart/util.hpp"
#include "popart/variablesettings.hpp"

class ErrorMessageContains {
private:
  const std::string expectedMessage_;

public:
  ErrorMessageContains(const std::string &expectedMessage)
      : expectedMessage_(expectedMessage) {}

  ErrorMessageContains(const char *expectedMessage)
      : expectedMessage_(expectedMessage) {}

  template <typename T> bool operator()(const T &error) {
    return std::string(error.what()).find(expectedMessage_) !=
           std::string::npos;
  }
};

struct TestConstructorWithReplicaGroupingData {
  const popart::VariableSettings settings;
  const popart::ReplicaGrouping expectedGrouping;
  const popart::VariableRetrievalMode expectedMode;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestConstructorWithReplicaGroupingData)

std::vector<TestConstructorWithReplicaGroupingData>
    testConstructorWithReplicaGroupingData{
        {popart::VariableSettings(8),
         popart::ReplicaGrouping(8),
         popart::VariableRetrievalMode::OnePerGroup},
        {popart::VariableSettings(popart::ReplicaGrouping(8)),
         popart::ReplicaGrouping(8),
         popart::VariableRetrievalMode::OnePerGroup},
        {popart::VariableSettings(8,
                                  popart::VariableRetrievalMode::AllReplicas),
         popart::ReplicaGrouping(8),
         popart::VariableRetrievalMode::AllReplicas},

        {popart::VariableSettings(popart::ReplicaGrouping(8),
                                  popart::VariableRetrievalMode::AllReplicas),
         popart::ReplicaGrouping(8),
         popart::VariableRetrievalMode::AllReplicas},
    };

BOOST_DATA_TEST_CASE(
    testConstructorWithReplicaGrouping,
    ::boost::unit_test::data::make(testConstructorWithReplicaGroupingData),
    testData) {
  const auto &settings         = testData.settings;
  const auto &expectedGrouping = testData.expectedGrouping;
  const auto &expectedMode     = testData.expectedMode;

  BOOST_CHECK_EQUAL(
      settings.getReplicaGrouping(expectedGrouping.getNumReplicas()),
      expectedGrouping);
  BOOST_CHECK_EQUAL(settings.getRetrievalMode(), expectedMode);
}

struct TestGetReplicaGroupingData {
  const popart::VariableSettings settings;
  const popart::ReplicaGrouping expectedGrouping;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestGetReplicaGroupingData)

std::vector<TestGetReplicaGroupingData> testGetReplicaGroupingData{
    {popart::VariableSettings(8), popart::ReplicaGrouping(8)},
    {popart::VariableSettings(popart::CommGroup()), popart::ReplicaGrouping(8)},
};

BOOST_DATA_TEST_CASE(testGetReplicaGrouping,
                     ::boost::unit_test::data::make(testGetReplicaGroupingData),
                     testData) {
  const auto &settings         = testData.settings;
  const auto &expectedGrouping = testData.expectedGrouping;

  BOOST_CHECK_EQUAL(settings.getReplicaGrouping(8), expectedGrouping);
}

// TODO(T62390): Enable this test.
BOOST_AUTO_TEST_CASE(testGetReplicaGroupingInvalid,
                     *boost::unit_test::disabled()) {
  const popart::VariableSettings settings{8};

  BOOST_CHECK_THROW(settings.getReplicaGrouping(4), popart::error);
}

struct TestGetSharedVariableDomainData {
  const popart::VariableSettings settings;
  const popart::CommGroup expectedCommGroup;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestGetSharedVariableDomainData)

std::vector<TestGetSharedVariableDomainData> testGetSharedVariableDomainData{
    {popart::VariableSettings(8), popart::CommGroup()},
    {popart::VariableSettings(popart::CommGroup()), popart::CommGroup()},
};

BOOST_DATA_TEST_CASE(
    testGetSharedVariableDomain,
    ::boost::unit_test::data::make(testGetSharedVariableDomainData),
    testData) {
  const auto &settings          = testData.settings;
  const auto &expectedCommGroup = testData.expectedCommGroup;

  BOOST_CHECK_EQUAL(settings.getSharedVariableDomain(), expectedCommGroup);
}

BOOST_AUTO_TEST_CASE(testGetSharedVariableDomainInvaid) {
  const popart::VariableSettings settings{popart::ReplicaGrouping(8, 2, 2)};
  const char *expectedMessage =
      "The 'ReplicaGrouping(numReplicas=8, stride=2, groupSize=2)' cannot be "
      "converted to a `popart::CommGroup`.";

  BOOST_CHECK_EXCEPTION(settings.getSharedVariableDomain(),
                        popart::error,
                        ErrorMessageContains(expectedMessage));
}

struct TestNumReplicasReturningVariableData {
  const popart::VariableRetrievalMode mode;
  const unsigned expected;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestNumReplicasReturningVariableData)

std::vector<TestNumReplicasReturningVariableData>
    testNumReplicasReturningVariableData{
        {popart::VariableRetrievalMode::OnePerGroup, 4},
        {popart::VariableRetrievalMode::AllReduceReplicas, 4},
        {popart::VariableRetrievalMode::AllReplicas, 8},
    };

BOOST_DATA_TEST_CASE(
    testNumReplicasReturningVariable,
    ::boost::unit_test::data::make(testNumReplicasReturningVariableData),
    testData) {
  const auto &mode     = testData.mode;
  const auto &expected = testData.expected;
  const popart::VariableSettings settings{popart::ReplicaGrouping(8, 4, 2),
                                          mode};

  BOOST_CHECK_EQUAL(settings.numReplicasReturningVariable(8), expected);
}

BOOST_AUTO_TEST_CASE(testGetGroupCount) {
  const popart::VariableSettings settings{popart::ReplicaGrouping(8, 4, 2)};

  BOOST_CHECK_EQUAL(settings.getGroupCount(8), 4);
}

BOOST_AUTO_TEST_CASE(testGetStride) {
  const popart::VariableSettings settings{popart::ReplicaGrouping(8, 4, 2)};

  BOOST_CHECK_EQUAL(settings.getStride(8), 4);
}

BOOST_AUTO_TEST_CASE(testGetRealGroupSize) {
  const popart::VariableSettings settings{popart::ReplicaGrouping(8, 4, 2)};

  BOOST_CHECK_EQUAL(settings.getRealGroupSize(8), 2);
}

BOOST_AUTO_TEST_CASE(testGetGroupRepresentative) {
  const popart::VariableSettings settings{popart::ReplicaGrouping(8, 2, 2)};

  BOOST_CHECK_EQUAL(settings.getGroupRepresentative(2), 4);
}

struct TestShapeOnReplicaData {
  const popart::ReplicaGrouping grouping;
  const popart::Shape shape;
  const popart::Shape expectedShape;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestShapeOnReplicaData)

std::vector<TestShapeOnReplicaData> testShapeOnReplicaData{
    {{popart::ReplicaGrouping(8)}, {}, {}},
    {{popart::ReplicaGrouping(8)}, {1}, {1}},
    {{popart::ReplicaGrouping(8)}, {4, 2}, {4, 2}},
    {{popart::ReplicaGrouping(8, 1, 4)}, {2}, {}},
    {{popart::ReplicaGrouping(8, 1, 4)}, {2, 3}, {3}},
};

BOOST_DATA_TEST_CASE(testShapeOnReplica,
                     ::boost::unit_test::data::make(testShapeOnReplicaData),
                     testData) {
  const auto &grouping      = testData.grouping;
  const auto &shape         = testData.shape;
  const auto &expectedShape = testData.expectedShape;
  const popart::VariableSettings settings{grouping};

  BOOST_CHECK_EQUAL(
      settings.shapeOnReplica(shape, grouping.getNumReplicas(), ""),
      expectedShape);
}

struct TestShapeOnReplicaInvalidData {
  const popart::ReplicaGrouping grouping;
  const popart::Shape shape;
  const char *expectedMessage;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestShapeOnReplicaInvalidData)

std::vector<TestShapeOnReplicaInvalidData> testShapeOnReplicaInvalidData{
    {{popart::ReplicaGrouping(8, 1, 4)},
     {},
     "Tensor 'foo' should have at least one dimension"},
    {{popart::ReplicaGrouping(8, 1, 4)},
     {3},
     "Return mismatch with possibly appended outer dimension"},
};

BOOST_DATA_TEST_CASE(
    testShapeOnReplicaInvalid,
    ::boost::unit_test::data::make(testShapeOnReplicaInvalidData),
    testData) {
  const auto &grouping        = testData.grouping;
  const auto &shape           = testData.shape;
  const auto &expectedMessage = testData.expectedMessage;
  const popart::VariableSettings settings{grouping};

  BOOST_CHECK_EXCEPTION(
      settings.shapeOnReplica(shape, grouping.getNumReplicas(), "foo"),
      popart::error,
      ErrorMessageContains(expectedMessage));
}

struct TestShapeOnHostData {
  const popart::ReplicaGrouping grouping;
  const popart::Shape shape;
  const popart::Shape expectedShape;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestShapeOnHostData)

std::vector<TestShapeOnHostData> testShapeOnHostData{
    {{popart::ReplicaGrouping(8)}, {}, {}},
    {{popart::ReplicaGrouping(8)}, {1}, {1}},
    {{popart::ReplicaGrouping(8)}, {4, 2}, {4, 2}},
    {{popart::ReplicaGrouping(8, 1, 4)}, {}, {2}},
    {{popart::ReplicaGrouping(8, 1, 4)}, {3}, {2, 3}},
};

BOOST_DATA_TEST_CASE(testShapeOnHost,
                     ::boost::unit_test::data::make(testShapeOnHostData),
                     testData) {
  const auto &grouping      = testData.grouping;
  const auto &shape         = testData.shape;
  const auto &expectedShape = testData.expectedShape;
  const popart::VariableSettings settings{grouping};

  BOOST_CHECK_EQUAL(settings.shapeOnHost(shape, grouping.getNumReplicas()),
                    expectedShape);
}

BOOST_AUTO_TEST_CASE(testGroups) {
  const popart::VariableSettings settings{popart::ReplicaGrouping(8, 2, 2)};
  const std::vector<std::vector<std::int64_t>> expectedGroups = {
      {0, 2}, {1, 3}, {4, 6}, {5, 7}};

  BOOST_CHECK_EQUAL(settings.groups(8), expectedGroups);
}

struct TestEqualityOperatorData {
  const popart::VariableSettings lhs;
  const popart::VariableSettings rhs;
  const bool expectedResult;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestEqualityOperatorData)

const std::vector<TestEqualityOperatorData> testEqualityOperatorData{
    {popart::CommGroup(), popart::CommGroup(), true},
    {popart::CommGroup(), popart::VariableSettings(8), false},
    {popart::VariableSettings(popart::CommGroup(),
                              popart::VariableRetrievalMode::OnePerGroup),
     popart::VariableSettings(popart::CommGroup(),
                              popart::VariableRetrievalMode::AllReplicas),
     false},
    {popart::VariableSettings(8), popart::VariableSettings(8), true},
    {popart::VariableSettings(8, popart::VariableRetrievalMode::OnePerGroup),
     popart::VariableSettings(8, popart::VariableRetrievalMode::AllReplicas),
     false},

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
