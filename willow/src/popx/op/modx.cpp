// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/mod.hpp>
#include <popart/popx/op/modx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

ModOpx::ModOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<ModOp>(op, {Onnx::Operators::Mod_10});
}

void ModOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::BinaryOpType::REMAINDER,
                           getInTensor(ModOp::getArg0InIndex()),
                           getInTensor(ModOp::getArg1InIndex()),
                           prog,
                           debugPrefix()));
}

namespace {
OpxCreator<ModOpx> modOpxCreator({Onnx::Operators::Mod_10});
OpxCreator<Opx> modArg0OpxCreator(
    Onnx::GradOperators::ModArg0Grad,
    "ModArg0Grad should be optimised out, \"ModArg0Grad\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
