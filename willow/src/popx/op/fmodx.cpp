// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <popops/ExprOp.hpp>
#include <popart/op/fmod.hpp>
#include <popart/popx/op/fmodx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

FmodOpx::FmodOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<FmodOp>(op, {Onnx::AiGraphcore::OpSet1::Fmod});
}

void FmodOpx::grow(snap::program::Sequence &prog) const {
  setOutTensor(FmodOp::getOutIndex(),
               snap::popops::map(graph(),
                                 popops::expr::BinaryOpType::REMAINDER,
                                 getInTensor(FmodOp::getArg0InIndex()),
                                 getInTensor(FmodOp::getArg1InIndex()),
                                 prog,
                                 debugContext()));
}

namespace {
OpxCreator<FmodOpx> fmodOpxCreator({Onnx::AiGraphcore::OpSet1::Fmod});
OpxCreator<PopOpx> fmodArg0OpxCreator(Onnx::GradOperators::FmodArg0Grad,
                                      "FmodArg0Grad should be optimised out, "
                                      "\"FmodArg0Grad\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
