// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/ir.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/getrandomseedx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

void GetRandomSeedOpx::grow(poplar::program::Sequence &prog) const {
  auto seed = getInTensor(op_p->getSeedInIndex());

  if (dv_p->ir().hasRandomOps()) {
    // Increment the seed
    auto one = getConst(seed.elementType(), {}, 1.0, "one");
    popops::addInPlace(graph(), seed, one, prog);
  }
  setOutTensor(GetRandomSeedOp::getUpdatedSeedOutIndex(), seed);
}

GetRandomSeedOpx::GetRandomSeedOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<GetRandomSeedOp>(op);
}

namespace {
OpxCreator<GetRandomSeedOpx>
    getRandomSeedOpxCreator(Onnx::CustomOperators::GetRandomSeed);
} // namespace

} // namespace popx
} // namespace popart
