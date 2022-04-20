// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <string>
#include <testutil/irquery/opstestwrapper.hpp>
#include <vector>

#include "testutil/irquery/irquery.hpp"

namespace popart {
class Ir;
class Op;

namespace irquery {

OpsTestWrapper::OpsTestWrapper(Ir &ir,
                               const std::vector<Op *> &ops,
                               const std::string &srcObjDescr_)
    : TestWrapper<std::vector<Op *>>{ir, ops}, srcObjDescr{srcObjDescr_} {}

} // namespace irquery
} // namespace popart
