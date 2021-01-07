// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/fmod.hpp>
#include <popart/popx/op/fmodx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

FmodOpx::FmodOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<FmodOp>(op, {Onnx::AiGraphcore::OpSet1::Fmod});
}

void FmodOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(FmodOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::BinaryOpType::REMAINDER,
                           getInTensor(FmodOp::getArg0InIndex()),
                           getInTensor(FmodOp::getArg1InIndex()),
                           prog,
                           debugContext()));
}

namespace {
OpxCreator<FmodOpx> fmodOpxCreator({Onnx::AiGraphcore::OpSet1::Fmod});
OpxCreator<Opx> fmodArg0OpxCreator(Onnx::GradOperators::FmodArg0Grad,
                                   "FmodArg0Grad should be optimised out, "
                                   "\"FmodArg0Grad\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
