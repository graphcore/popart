// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/sign.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/signx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Zero.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

SignOpx::SignOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SignOp>(op, {Onnx::Operators::Sign_9});
}

void SignOpx::grow(poplar::program::Sequence &prog) const {

  setOutTensor(SignOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::UnaryOpType::SIGNUM,
                           getInTensor(SignOp::getInIndex()),
                           prog,
                           debugPrefix()));
}

SignGradOpx::SignGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SignGradOp>(op, Onnx::GradOperators::SignGrad);
}

void SignGradOpx::grow(poplar::program::Sequence &) const {

  auto outTensor = getConst(popType(outInfo(SignGradOp::getOutIndex())),
                            outInfo(SignGradOp::getOutIndex()).shape_szt(),
                            0,
                            debugPrefix("zeros"));

  setOutTensor(SignGradOp::getOutIndex(), outTensor);
}

namespace {
OpxCreator<SignOpx> signOpxCreator(Onnx::Operators::Sign_9);
OpxCreator<SignGradOpx> signGradOpxCreator(Onnx::GradOperators::SignGrad);
} // namespace

} // namespace popx
} // namespace popart
