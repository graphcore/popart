// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE ReplicatedTensorMutableVoidInfoVerifier

#include <boost/test/unit_test.hpp>
#include <stdexcept>
#include <string>
#include <vector>
#include <popart/commgroup.hpp>
#include <popart/ir.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/variablesettings.hpp>

#include "popart/datatype.hpp"

namespace popart {
class internal_error;
} // namespace popart

using namespace popart;

/**
 * This test runs all the permutations of
 * VariableSettings from the Config's object,
 * and the multiple different replication factors
 * to verify that the function in question:
 * verifyMutableVoidInfo,
 * correctly assertains wether or not the MutableVoidInfo
 * of a MutableVoidData points to an appropriatly sized
 * buffer.
 */
BOOST_AUTO_TEST_CASE(ReplicatedTensorMutableVoidInfoVerifier) {
  popart::Ir ir;
  auto &graph = ir.getMainGraph();

  auto one = VariableRetrievalMode::OnePerGroup;
  auto all = VariableRetrievalMode::AllReplicas;

  std::vector<VariableSettings> configs = {
      VariableSettings(CommGroup(CommGroupType::All, 0), one),
      VariableSettings(CommGroup(CommGroupType::None, 0), one),
      VariableSettings(CommGroup(CommGroupType::All, 0), all),
      VariableSettings(CommGroup(CommGroupType::None, 0), all),
      VariableSettings(CommGroup(CommGroupType::Consecutive, 2), one),
      VariableSettings(CommGroup(CommGroupType::Orthogonal, 2), all)};

  std::vector<long int> base{3, 4};

  for (VariableSettings vs : configs) {
    auto id = std::string("ofVS[cg(") +
              static_cast<char>(vs.getSharedVariableDomain().type) +
              std::string(", ") +
              static_cast<char>(vs.getSharedVariableDomain().replicaGroupSize) +
              std::string(") ") + static_cast<char>(vs.getRetrievalMode()) +
              std::string("]");

    Tensor t(id, vs, graph);
    t.info = TensorInfo(DataType::FLOAT, base);

    std::vector<int> replications{1, 2, 4};

    // with replication factor
    for (int &replication : replications) {
      int r = (int)vs.numReplicasReturningVariable(replication);
      std::vector<long int> rbase = {r, 3, 4};
      std::vector<long int> fail  = {r + 1, 3, 4};
      std::vector<long int> shape = rbase[0] == 1 ? base : rbase;

      TensorInfo ti_base(DataType::FLOAT, base);
      TensorInfo ti_fail(DataType::FLOAT, fail);
      TensorInfo ti_shape(DataType::FLOAT, shape);

      // Check that the correct shape works (will throw error if not)
      t.verifyMutableVoidInfo(ti_shape, replication);

      // Check that incorrect shape fails
      BOOST_CHECK_EXCEPTION(
          t.verifyMutableVoidInfo(ti_fail, replication),
          popart::internal_error,
          [](const popart::internal_error &err) { return true; });

      if (r == 1) {
        // If it returns 1, verify it supports the base shape
        t.verifyMutableVoidInfo(ti_base, replication);
      } else {
        // If it returns multiple, verify it doesn't support the base
        BOOST_CHECK_EXCEPTION(
            t.verifyMutableVoidInfo(ti_base, replication),
            popart::internal_error,
            [](const std::runtime_error &err) { return true; });
      }
    }
  }
}
