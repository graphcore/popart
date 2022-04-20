// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/getrandomseedx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
namespace popx {

void GetRandomSeedOpx::grow(snap::program::Sequence &prog) const {
  auto seed = getInTensor(op_p->getSeedInIndex());

  // Increment the seed
  auto one = getConst(seed.elementType(), {1}, 1.0, "one");
  // The LHS of the seed is offset by the replication index when loaded onto the
  // device, see IrLowering::initRandomSeed(). Incrementing by replicationFactor
  // ensures no overlap in the LHS of the seed between replicas
  auto grf = getConst(seed.elementType(),
                      {1},
                      dv_p->lowering().getGlobalReplicationFactor(),
                      "globalReplicationFactor");
  snap::popops::addInPlace(graph(),
                           seed,
                           snap::concat({grf, one}),
                           prog,
                           debugContext("RandomSeedIncrement"));

  setOutTensor(GetRandomSeedOp::getUpdatedSeedOutIndex(), seed);
}

GetRandomSeedOpx::GetRandomSeedOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<GetRandomSeedOp>(op);
}

namespace {
OpxCreator<GetRandomSeedOpx>
    getRandomSeedOpxCreator(Onnx::CustomOperators::GetRandomSeed);
} // namespace

} // namespace popx
} // namespace popart
