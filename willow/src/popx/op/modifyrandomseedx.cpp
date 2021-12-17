// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprand/RandomGen.hpp>
#include <popart/ir.hpp>
#include <popart/op/modifyrandomseed.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/modifyrandomseedx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

#include <cstdint>
#include <limits>

namespace pe = popops::expr;

namespace popart {
namespace popx {

ModifyRandomSeedOpx::ModifyRandomSeedOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ModifyRandomSeedOp>(op);
}

void ModifyRandomSeedOpx::grow(snap::program::Sequence &prog) const {
  auto inSeed   = getInTensor(op_p->getSeedInIndex()).getPoplarTensor();
  auto modifier = getInTensor(ModifyRandomSeedOp::getSeedModifierInIndex())
                      .getPoplarTensor();

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
  auto rPlusMod = popops::add(graph().getPoplarGraph(),
                              inSeedR,
                              modifier,
                              prog.getPoplarSequence(),
                              debugContext("preSeed"));
  auto metaSeed = poplar::concat({inSeedL, rPlusMod});
  // Calculate randint(R + modifier).
  auto outSeedRAsInt = poprand::uniform(
      graph().getPoplarGraph(),
      &metaSeed,
      0u,
      inSeedR,
      poplar::INT, // unsigned int not supported.
      static_cast<double>(std::numeric_limits<std::int32_t>::min()),
      static_cast<double>(std::numeric_limits<std::int32_t>::max()),
      prog.getPoplarSequence(),
      debugContext("pickSeed"));
  // Map randint(R + modifier) to UNSIGNED INT.
  auto outSeedR = popops::map(graph().getPoplarGraph(),
                              pe::Cast(pe::_1, poplar::UNSIGNED_INT),
                              {outSeedRAsInt},
                              prog.getPoplarSequence(),
                              debugContext("castToUint"));
  // Concatenate outSeed.
  auto outSeedUncopied = poplar::concat({inSeedL, outSeedR});
  // Copy.
  auto outSeed = cloneNcopy(prog, snap::Tensor{outSeedUncopied, graph()});

  setOutTensor(ModifyRandomSeedOp::getModifiedSeedOutIndex(), outSeed);
}

namespace {
OpxCreator<ModifyRandomSeedOpx>
    ModifyRandomSeedOpxCreator(Onnx::CustomOperators::ModifyRandomSeed);
} // namespace

} // namespace popx
} // namespace popart
