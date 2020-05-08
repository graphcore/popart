// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

CollectivesBaseOp::CollectivesBaseOp(const OperatorIdentifier &_opid,
                                     const Op::Settings &settings_)
    : Op(_opid, settings_) {}

} // namespace popart
