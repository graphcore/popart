// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <limits>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <poprand/RandomGen.hpp>
#include <popart/op/modifyrandomseed.hpp>
#include <popart/popx/op/modifyrandomseedx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
namespace popx {
class Devicex;

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
  // is used to increment the second value and this value is used as a seed
  // to produce a new random seed:
  //   outSeed = [L, randint(R + modifier)].

  auto inSeedL = inSeed.slice(0, 1);
  auto inSeedR = inSeed.slice(1, 2);

  // Calculate R + modifier.
  auto rPlusMod =
      popops::add(graph(), inSeedR, modifier, prog, debugContext("preSeed"));
  auto metaSeed = poplar::concat({inSeedL, rPlusMod});
  // Calculate randint(R + modifier).
  auto outSeedRAsInt = poprand::uniform(
      graph(),
      &metaSeed,
      0u,
      inSeedR,
      poplar::INT, // unsigned int not supported.
      static_cast<double>(std::numeric_limits<std::int32_t>::min()),
      static_cast<double>(std::numeric_limits<std::int32_t>::max()),
      prog,
      debugContext("pickSeed"));
  // Map randint(R + modifier) to UNSIGNED INT.
  auto outSeedR = popops::map(graph(),
                              pe::Cast(pe::_1, poplar::UNSIGNED_INT),
                              {outSeedRAsInt},
                              prog,
                              debugContext("castToUint"));
  // Concatenate outSeed.
  auto outSeedUncopied = poplar::concat({inSeedL, outSeedR});
  // Copy.
  auto outSeed = cloneNcopy(prog, outSeedUncopied);

  setOutTensor(ModifyRandomSeedOp::getModifiedSeedOutIndex(), outSeed);
}

namespace {
OpxCreator<ModifyRandomSeedOpx>
    ModifyRandomSeedOpxCreator(Onnx::CustomOperators::ModifyRandomSeed);
} // namespace

} // namespace popx
} // namespace popart
