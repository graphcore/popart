// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <testutil/irquery/graphtestwrapper.hpp>
#include <testutil/irquery/irtestwrapper.hpp>
#include <testutil/irquery/testfailuretriggerer.hpp>

namespace popart {
namespace irquery {

IrTestWrapper::IrTestWrapper(Ir &ir)
    : TestWrapper<std::reference_wrapper<Ir>>{ir, ir} {}

nonstd::optional<GraphTestWrapper> IrTestWrapper::hasGraph(GraphId id,
                                                           Require testReq) {
  bool result = false;
  nonstd::optional<GraphTestWrapper> userValue;

  if (ir.get().hasGraph(id)) {
    result    = true;
    userValue = GraphTestWrapper{ir, id};
  }

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected to find graph with ID '" << id << "' in IR";

    triggerer->trigger(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect to find graph with ID '" << id << "' in IR";

    triggerer->trigger(ss.str());
  }

  return userValue;
}

} // namespace irquery
} // namespace popart