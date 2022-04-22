// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TensorTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <memory>
#include <testutil/test_graphs/graph_test_models.hpp>
#include <vector>
#include <popart/aliasesmap.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/tensor.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/region.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/tensors.hpp"
#include "popart/util.hpp"

namespace popart {

class Op;
} // namespace popart

using namespace popart;
using namespace graphutils;

BOOST_AUTO_TEST_CASE(ModifiedRegionsByOpsTest0) {
  GraphTestModel2 model;

  Tensor *t0    = model.getIr().getMainGraph().getTensors().get("t0");
  auto schedule = model.getIr().getMainGraph().getOpSchedule(
      {}, RequireOptimalSchedule::No);

  AliasesMap aliasesMap{model.getIr().getMainGraph()};
  auto &aliases = aliasesMap.getAliases(model.getIr().getMainGraph());

  {
    // Check the full schedule
    auto regions = t0->modifiedRegionsByOps(schedule, aliases);

    logging::debug("Regions: {}", regions);

    BOOST_REQUIRE_EQUAL(regions.size(), 1);
    BOOST_CHECK_EQUAL(regions, view::Regions{view::Region({0, 0}, {4, 4})});
    BOOST_CHECK_EQUAL(regions.front().getAccessType(),
                      view::AccessType::ReadWrite);
  }

  {
    // Only check consumers of t3 (partial alias of t0)
    auto consumers = model.getIr().getTensor("t3")->consumers.getOps();
    std::vector<Op *> opsToTest;
    for (Op *op : schedule) {
      if (std::find(consumers.begin(), consumers.end(), op) !=
          consumers.end()) {
        opsToTest.push_back(op);
      }
    }

    BOOST_REQUIRE_EQUAL(opsToTest.size(), 2);

    {
      auto regions = t0->modifiedRegionsByOps(opsToTest, aliases);

      logging::debug("Regions: {}", regions);

      BOOST_REQUIRE_EQUAL(regions.size(), 1);
      BOOST_CHECK_EQUAL(regions, view::Regions{view::Region({2, 0}, {4, 4})});
      BOOST_CHECK_EQUAL(regions.front().getAccessType(),
                        view::AccessType::ReadWrite);
    }
  }

  {
    // Only check consumers of t2 (partial alias of t0)
    auto consumers = model.getIr().getTensor("t2")->consumers.getOps();
    std::vector<Op *> opsToTest;
    for (Op *op : schedule) {
      if (std::find(consumers.begin(), consumers.end(), op) !=
          consumers.end()) {
        opsToTest.push_back(op);
      }
    }

    BOOST_REQUIRE_EQUAL(opsToTest.size(), 1);

    {
      auto regions = t0->modifiedRegionsByOps(opsToTest, aliases);

      logging::debug("Regions: {}", regions);

      BOOST_REQUIRE_EQUAL(regions.size(), 1);
      BOOST_CHECK_EQUAL(regions, view::Regions{view::Region({0, 0}, {2, 4})});
      BOOST_CHECK_EQUAL(regions.front().getAccessType(),
                        view::AccessType::Write);
    }
  }
}
