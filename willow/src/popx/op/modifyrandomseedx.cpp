// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/ir.hpp>
#include <popart/op/modifyrandomseed.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/modifyrandomseedx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

ModifyRandomSeedOpx::ModifyRandomSeedOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ModifyRandomSeedOp>(op);
}

void ModifyRandomSeedOpx::grow(poplar::program::Sequence &prog) const {
  auto inSeed   = getInTensor(op_p->getSeedInIndex());
  auto modifier = getInTensor(ModifyRandomSeedOp::getSeedModifierInIndex());

  // The seed supplied to poprand is defined as a pair of uint32 values,
  //   inSeed = [L, R].
  //
  // For independent reproducible random number streams, a modifier
  // is used to increment the second value to define the output seed:
  //   outSeed = [L, R + modifier].

  auto inSeedL         = inSeed.slice(0, 1);
  auto inSeedR         = inSeed.slice(1, 2);
  auto inSeedRMod      = popops::add(graph(), inSeedR, modifier, prog);
  auto outSeedUncopied = poplar::concat({inSeedL, inSeedRMod});
  auto outSeed         = cloneNcopy(prog, outSeedUncopied);

  setOutTensor(ModifyRandomSeedOp::getModifiedSeedOutIndex(), outSeed);
}

namespace {
OpxCreator<ModifyRandomSeedOpx>
    ModifyRandomSeedOpxCreator(Onnx::CustomOperators::ModifyRandomSeed);
} // namespace

} // namespace popx
} // namespace popart
