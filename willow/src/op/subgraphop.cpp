// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/op/subgraphop.hpp>
#include <popart/opserialiser.hpp>
#include <popart/scope.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

SubgraphOp::SubgraphOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

} // namespace popart
