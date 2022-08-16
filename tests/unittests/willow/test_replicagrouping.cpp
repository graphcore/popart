// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_replicagrouping

#include <algorithm>
#include <string>
#include <vector>

#include "boost/test/data/test_case.hpp"
#include "boost/test/unit_test.hpp"
#include "popart/error.hpp"
#include "popart/replicagrouping.hpp"
#include "popart/util.hpp"

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

struct TestConstructorArgValidationData {
  const std::vector<unsigned> args;
  const char *expectedMessage;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestConstructorArgValidationData)

const std::vector<TestConstructorArgValidationData>
    testConstructorArgValidationData{
        {{0, 1, 1},
         "The number of replicas in a `popart::ReplicaGrouping` must be a "
         "positive integer."},
        {{1, 0, 1},
         "The stride in a `popart::ReplicaGrouping` must be a positive "
         "integer."},
        {{1, 1, 0},
         "The group size in a `popart::ReplicaGrouping` must be a positive "
         "integer."},
        {{4, 3, 1},
         "The number of replicas in a `popart::ReplicaGrouping` must be "
         "divisible by the product of the stride and the group size."},
        {{4, 1, 3},
         "The number of replicas in a `popart::ReplicaGrouping` must be "
         "divisible by the product of the stride and the group size."},
        {{12, 4, 2},
         "The number of replicas in a `popart::ReplicaGrouping` must be "
         "divisible by the product of the stride and the group size."},
        {{4, 4, 2},
         "The number of replicas in a `popart::ReplicaGrouping` must be "
         "divisible by the product of the stride and the group size."},
    };

BOOST_DATA_TEST_CASE(
    testConstructorArgValidation,
    ::boost::unit_test::data::make(testConstructorArgValidationData),
    testData) {
  const auto &args            = testData.args;
  const auto &expectedMessage = testData.expectedMessage;

  BOOST_CHECK_EXCEPTION(popart::ReplicaGrouping(args[0], args[1], args[2]),
                        popart::error,
                        ErrorMessageContains(expectedMessage));
}

BOOST_AUTO_TEST_CASE(testGetNumReplicas) {
  BOOST_CHECK_EQUAL(popart::ReplicaGrouping(8, 4, 2).getNumReplicas(), 8);
}

BOOST_AUTO_TEST_CASE(testGetStride) {
  BOOST_CHECK_EQUAL(popart::ReplicaGrouping(8, 2, 1).getStride(), 1);
  BOOST_CHECK_EQUAL(popart::ReplicaGrouping(8, 4, 2).getStride(), 4);
}

BOOST_AUTO_TEST_CASE(testGetGroupSize) {
  BOOST_CHECK_EQUAL(popart::ReplicaGrouping(8, 4, 2).getGroupSize(), 2);
}

BOOST_AUTO_TEST_CASE(testGetNumGroups) {
  BOOST_CHECK_EQUAL(popart::ReplicaGrouping(8, 1, 2).getNumGroups(), 4);
}

struct TestGetGroupAtData {
  const popart::ReplicaGrouping grouping;
  const std::vector<unsigned> expectedResult;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestGetGroupAtData)

const std::vector<TestGetGroupAtData> testGetGroupAtData{
    {{1, 1, 1}, {0}},
    {{2, 1, 1}, {0, 1}},
    {{2, 2, 1}, {0, 1}},
    {{2, 1, 2}, {0, 0}},
    {{3, 1, 1}, {0, 1, 2}},
    {{3, 3, 1}, {0, 1, 2}},
    {{3, 1, 3}, {0, 0, 0}},
    {{4, 1, 1}, {0, 1, 2, 3}},
    {{4, 2, 1}, {0, 1, 2, 3}},
    {{4, 4, 1}, {0, 1, 2, 3}},
    {{4, 1, 2}, {0, 0, 1, 1}},
    {{4, 2, 2}, {0, 1, 0, 1}},
    {{4, 1, 4}, {0, 0, 0, 0}},
};

BOOST_DATA_TEST_CASE(testGetGroupAt,
                     ::boost::unit_test::data::make(testGetGroupAtData),
                     testData) {
  const auto &grouping       = testData.grouping;
  const auto &expectedResult = testData.expectedResult;

  std::vector<unsigned> result(grouping.getNumReplicas());
  std::generate(
      result.begin(), result.end(), [&grouping, replica = 0]() mutable {
        return grouping.getGroupAt(replica++);
      });

  BOOST_CHECK_EQUAL(result, expectedResult);
}

BOOST_AUTO_TEST_CASE(testGetGroupAtInvalidArg) {
  const popart::ReplicaGrouping grouping{1, 1, 1};
  const auto expectedMessage =
      "The requested replica index is outside the valid range for "
      "'ReplicaGrouping(numReplicas=1, stride=1, groupSize=1)'.";

  BOOST_CHECK_EXCEPTION(grouping.getGroupAt(1),
                        popart::error,
                        ErrorMessageContains(expectedMessage));
}

BOOST_AUTO_TEST_CASE(getIndexInGroupAt) {
  const popart::ReplicaGrouping grouping{18, 2, 3};
  const std::vector<unsigned> expectedResult = {
      0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2};

  std::vector<unsigned> result(grouping.getNumReplicas());
  std::generate(
      result.begin(), result.end(), [&grouping, replica = 0]() mutable {
        return grouping.getIndexInGroupAt(replica++);
      });
  BOOST_CHECK_EQUAL(result, expectedResult);
}

struct TestGetReplicasAtData {
  const popart::ReplicaGrouping grouping;
  const std::vector<std::vector<unsigned>> expectedResult;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestGetReplicasAtData)

const std::vector<TestGetReplicasAtData> testGetReplicasAtData{
    {{1, 1, 1}, {{0}}},
    {{2, 1, 1}, {{0}, {1}}},
    {{2, 2, 1}, {{0}, {1}}},
    {{2, 1, 2}, {{0, 1}}},
    {{3, 1, 1}, {{0}, {1}, {2}}},
    {{3, 3, 1}, {{0}, {1}, {2}}},
    {{3, 1, 3}, {{0, 1, 2}}},
    {{4, 1, 1}, {{0}, {1}, {2}, {3}}},
    {{4, 2, 1}, {{0}, {1}, {2}, {3}}},
    {{4, 4, 1}, {{0}, {1}, {2}, {3}}},
    {{4, 1, 2}, {{0, 1}, {2, 3}}},
    {{4, 2, 2}, {{0, 2}, {1, 3}}},
    {{4, 1, 4}, {{0, 1, 2, 3}}},
};

BOOST_DATA_TEST_CASE(testGetReplicaAt,
                     ::boost::unit_test::data::make(testGetReplicasAtData),
                     testData) {
  const auto &grouping       = testData.grouping;
  const auto &expectedResult = testData.expectedResult;

  std::vector<std::vector<unsigned>> result(grouping.getNumGroups());
  std::generate(result.begin(), result.end(), [&grouping, group = 0]() mutable {
    std::vector<unsigned> replicas(grouping.getGroupSize());
    std::generate(replicas.begin(),
                  replicas.end(),
                  [&grouping, &group, index = 0]() mutable {
                    return grouping.getReplicaAt(group, index++);
                  });
    group++;
    return replicas;
  });

  BOOST_CHECK_EQUAL(result, expectedResult);
}

struct TestGetReplicasAtInvalidArgsData {
  const unsigned group;
  const unsigned index;
  const std::string expectedMessage;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestGetReplicasAtInvalidArgsData)

const std::vector<TestGetReplicasAtInvalidArgsData>
    testGetReplicasAtInvalidArgsData{
        {1,
         0,
         "The requested group index is outside the valid range for "
         "'ReplicaGrouping(numReplicas=1, stride=1, groupSize=1)'."},
        {0,
         1,
         "The requested index is outside the valid range for "
         "'ReplicaGrouping(numReplicas=1, stride=1, groupSize=1)'."},
    };

BOOST_DATA_TEST_CASE(
    testGetReplicaAtInvalidArgs,
    ::boost::unit_test::data::make(testGetReplicasAtInvalidArgsData),
    testData) {
  const popart::ReplicaGrouping grouping{1, 1, 1};
  const auto &group           = testData.group;
  const auto &index           = testData.index;
  const auto &expectedMessage = testData.expectedMessage;

  BOOST_CHECK_EXCEPTION(grouping.getReplicaAt(group, index),
                        popart::error,
                        ErrorMessageContains(expectedMessage));
}

BOOST_DATA_TEST_CASE(testGetReplicasAt,
                     ::boost::unit_test::data::make(testGetReplicasAtData),
                     testData) {
  const auto &grouping       = testData.grouping;
  const auto &expectedResult = testData.expectedResult;

  std::vector<std::vector<unsigned>> result(grouping.getNumGroups());
  std::generate(result.begin(), result.end(), [&grouping, group = 0]() mutable {
    return grouping.getReplicasAt(group++);
  });

  BOOST_CHECK_EQUAL(result, expectedResult);
}

BOOST_AUTO_TEST_CASE(testGetReplicasAtInvalidArg) {
  const popart::ReplicaGrouping grouping{1, 1, 1};
  const auto expectedMessage =
      "The requested group index is outside the valid range for "
      "'ReplicaGrouping(numReplicas=1, stride=1, groupSize=1)'.";

  BOOST_CHECK_EXCEPTION(grouping.getReplicasAt(1),
                        popart::error,
                        ErrorMessageContains(expectedMessage));
}

struct TestGetTransposeData {
  const popart::ReplicaGrouping original;
  const popart::ReplicaGrouping transposed;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestGetTransposeData)

const std::vector<TestGetTransposeData> testGetTransposeData{
    // [[0]].T
    // = [[0]]
    {{1, 1, 1}, {1, 1, 1}},
    // [[0], [1], ..., [16]].T
    // = [[0, 1, ..., 16]]
    {{16, 1, 1}, {16, 1, 16}},
    {{16, 16, 1}, {16, 1, 16}},
    // [[0, 1, 2, 3], ..., [12, 13, 14, 15]].T
    // = [[0, 4, 8, 12], ..., [3, 7, 11, 15]]
    {{16, 1, 4}, {16, 4, 4}},
    // [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]].T
    // = [[0, 1], ..., [14, 15]]
    {{16, 2, 8}, {16, 1, 2}},
    // [[0, 8], ..., [7, 15]].T
    // = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    {{16, 8, 2}, {16, 1, 8}},
};

BOOST_DATA_TEST_CASE(testGetTranspose,
                     ::boost::unit_test::data::make(testGetTransposeData),
                     testData) {
  const auto &original   = testData.original;
  const auto &transposed = testData.transposed;

  BOOST_CHECK_EQUAL(original.getTranspose(), transposed);
  BOOST_CHECK_EQUAL(transposed.getTranspose(), original);
}

BOOST_AUTO_TEST_CASE(testGetTransposeInvalid) {
  const popart::ReplicaGrouping grouping{12, 3, 2};
  const auto expectedMessage =
      "The transpose of 'ReplicaGrouping(numReplicas=12, stride=3, "
      "groupSize=2)' cannot be represented as a `popart::ReplicaGrouping`.";

  BOOST_CHECK_EXCEPTION(grouping.getTranspose(),
                        popart::error,
                        ErrorMessageContains(expectedMessage));
}

BOOST_AUTO_TEST_CASE(testStr) {
  const popart::ReplicaGrouping grouping{4, 1, 2};
  const std::string expectedStr =
      "ReplicaGrouping(numReplicas=4, stride=1, groupSize=2)";

  BOOST_CHECK_EQUAL(grouping.str(), expectedStr);
}

struct TestEqualityOperatorData {
  const popart::ReplicaGrouping lhs;
  const popart::ReplicaGrouping rhs;
  const bool expectedResult;
};
BOOST_TEST_DONT_PRINT_LOG_VALUE(TestEqualityOperatorData)

const std::vector<TestEqualityOperatorData> testEqualityOperatorData{
    {{4, 2, 1}, {4, 2, 1}, true},
    {{4, 2, 1}, {4, 1, 1}, true},
    {{8, 2, 1}, {4, 2, 1}, false},
    {{4, 2, 2}, {4, 2, 1}, false},
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

BOOST_AUTO_TEST_CASE(testStreamInsertionOperator) {
  const popart::ReplicaGrouping grouping{4, 1, 2};
  const std::string expectedStr =
      "ReplicaGrouping(numReplicas=4, stride=1, groupSize=2)";

  std::stringstream ss;
  ss << grouping;

  BOOST_CHECK_EQUAL(ss.str(), expectedStr);
}
