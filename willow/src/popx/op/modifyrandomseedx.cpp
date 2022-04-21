// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprand/RandomGen.hpp>
#include <popart/ir.hpp>
#include <popart/op/modifyrandomseed.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/modifyrandomseedx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <snap/popops/ElementWise.hpp>
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
  auto rPlusMod = snap::Tensor{popops::add(graph().getPoplarGraph(),
                                           inSeedR.getPoplarTensor(),
                                           modifier.getPoplarTensor(),
                                           prog.getPoplarSequence(),
                                           debugContext("preSeed")),
                               graph()};
  auto metaSeed = snap::concat({inSeedL, rPlusMod});
  // Calculate randint(R + modifier).
  auto outSeedRAsInt = snap::Tensor{
      poprand::uniform(
          graph().getPoplarGraph(),
          &metaSeed.getPoplarTensor(),
          0u,
          inSeedR.getPoplarTensor(),
          poplar::INT, // unsigned int not supported.
          static_cast<double>(std::numeric_limits<std::int32_t>::min()),
          static_cast<double>(std::numeric_limits<std::int32_t>::max()),
          prog.getPoplarSequence(),
          debugContext("pickSeed")),
      graph()};
  // Map randint(R + modifier) to UNSIGNED INT.
  auto outSeedR = snap::popops::map(graph(),
                                    pe::Cast(pe::_1, poplar::UNSIGNED_INT),
                                    {outSeedRAsInt},
                                    prog,
                                    debugContext("castToUint"));
  // Concatenate outSeed.
  auto outSeedUncopied = snap::concat({inSeedL, outSeedR});
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
