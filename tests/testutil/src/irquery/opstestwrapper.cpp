// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op.hpp>

#include <testutil/irquery/opstestwrapper.hpp>
#include <testutil/irquery/testfailuretriggerer.hpp>

namespace popart {
namespace irquery {

OpsTestWrapper::OpsTestWrapper(Ir &ir,
                               const std::vector<Op *> &ops,
                               const std::string &srcObjDescr_)
    : TestWrapper<std::vector<Op *>>{ir, ops}, srcObjDescr{srcObjDescr_} {}

} // namespace irquery
} // namespace popart
