// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_OPS_TEST_WRAPPER_IMPL_HPP
#define TEST_UTILS_IR_QUERY_OPS_TEST_WRAPPER_IMPL_HPP

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <testutil/irquery/opstestwrapper.hpp>
#include <testutil/irquery/optestwrapper.hpp>
#include <testutil/irquery/testfailuretriggerer.hpp>
#include <vector>

#include "popart/vendored/optional.hpp"
#include "testutil/irquery/require.hpp"

namespace popart {
class Op;

namespace irquery {

template <typename OP, typename enableif>
nonstd::optional<OpTestWrapper<OP, enableif>>
OpsTestWrapper::hasOp(OpPred<OP> opPred, Require testReq) {
  std::vector<Op *> &ops = wrappedObj;

  bool result = false;
  nonstd::optional<OpTestWrapper<OP>> userValue;

  auto opIt = std::find_if(ops.begin(), ops.end(), [&](auto &op) {
    if (OP *castedOp = dynamic_cast<OP *>(op)) {
      OpTestWrapper<OP> otw{ir, castedOp};
      return opPred(otw);
    } else {
      return false;
    }
  });

  if (opIt != ops.end()) {
    userValue = OpTestWrapper<OP>{ir, dynamic_cast<OP *>(*opIt)};
    result    = true;
  }

  if (testReq == Require::MustBeTrue && !result) {

    std::stringstream ss;
    ss << "Expected to find an op in " << srcObjDescr
       << " that matches predicate";

    triggerer->trigger(ss.str());

  } else if (testReq == Require::MustBeFalse && result) {

    std::stringstream ss;
    ss << "Did not expect to find an op in " << srcObjDescr
       << " that matches predicate";

    triggerer->trigger(ss.str());
  }

  return userValue;
}

template <typename OP, typename enableif>
nonstd::optional<OpTestWrapper<OP, enableif>>
OpsTestWrapper::hasOp(Require testReq) {
  auto tautology = [](OpTestWrapper<OP> &op) -> bool { return true; };
  return hasOp<OP>(tautology, testReq);
}

} // namespace irquery
} // namespace popart

#endif
