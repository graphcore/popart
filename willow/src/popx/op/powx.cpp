// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/op/pow.hpp>
#include <popart/popx/op/powx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

PowOpx::PowOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<PowOp>(op, {Onnx::Operators::Pow_1, Onnx::Operators::Pow_7});
}

void PowOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::BinaryOpType::POWER,
                           getInTensor(PowOp::getArg0InIndex()),
                           getInTensor(PowOp::getArg1InIndex()),
                           prog,
                           debugPrefix()));
}

namespace {
OpxCreator<PowOpx> powOpxCreator({Onnx::Operators::Pow_1,
                                  Onnx::Operators::Pow_7});
OpxCreator<Opx> powArg0OpxCreator(Onnx::GradOperators::PowArg0Grad,
                                  "PowArg0Grad should be optimised out, "
                                  "\"PowArg0GradOp\" pattern is required");
OpxCreator<Opx> powArg1OpxCreator(Onnx::GradOperators::PowArg1Grad,
                                  "PowArg1Grad should be optimised out, "
                                  "\"PowArg1GradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
