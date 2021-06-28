// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <popops/ElementWise.hpp>
#include <popart/op/sign.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/signx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

snap::Tensor SignComputex::outplace(poplar::program::Sequence &prog,
                                    snap::Graph &graph,
                                    const snap::Tensor &tensor,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &s) const {

  return snap::Tensor{popops::map(graph.getPoplarGraph(),
                                  popops::expr::UnaryOpType::SIGNUM,
                                  tensor.getPoplarTensor(),
                                  prog,
                                  {dnai, s}),
                      graph};
}

void SignComputex::inplace(poplar::program::Sequence &prog,
                           snap::Graph &graph,
                           const snap::Tensor &tensor,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  popops::mapInPlace(graph.getPoplarGraph(),
                     popops::expr::UnaryOpType::SIGNUM,
                     tensor.getPoplarTensor(),
                     prog,
                     {dnai, s});
}

SignOpx::SignOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, SignComputex::get()) {
  verifyOp<SignOp>(op, {Onnx::Operators::Sign_9});
}

SignInplaceOpx::SignInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, SignComputex::get()) {
  verifyOp<SignInplaceOp>(op, Onnx::CustomOperators::SignInplace);
}

namespace {
OpxCreator<SignOpx> signOpxCreator(Onnx::Operators::Sign_9);
OpxCreator<SignInplaceOpx>
    signInplaceOpxCreator(Onnx::CustomOperators::SignInplace);
} // namespace

} // namespace popx
} // namespace popart
