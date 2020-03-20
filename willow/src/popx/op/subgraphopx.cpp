// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op/call.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/subgraphopx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

SubgraphOpx::SubgraphOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

} // namespace popx
} // namespace popart
