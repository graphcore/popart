// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <popops/ElementWise.hpp>
#include <popart/op/sign.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/signx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

poplar::Tensor SignComputex::outplace(poplar::program::Sequence &prog,
                                      poplar::Graph &graph,
                                      const poplar::Tensor &tensor,
                                      const std::string &s) const {

  return popops::map(graph, popops::expr::UnaryOpType::SIGNUM, tensor, prog, s);
}

void SignComputex::inplace(poplar::program::Sequence &prog,
                           poplar::Graph &graph,
                           const poplar::Tensor &tensor,
                           const std::string &s) const {

  popops::mapInPlace(graph, popops::expr::UnaryOpType::SIGNUM, tensor, prog, s);
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
