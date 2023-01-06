// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <string>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/signx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace poplar {
class Graph;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;
class SignInplaceOp;
class SignOp;

namespace popx {
class Devicex;

poplar::Tensor SignComputex::outplace(poplar::program::Sequence &prog,
                                      poplar::Graph &graph,
                                      const poplar::Tensor &tensor,
                                      const poplar::DebugNameAndId &dnai,
                                      const std::string &s) const {

  return popops::map(
      graph, popops::expr::UnaryOpType::SIGNUM, tensor, prog, {dnai, s});
}

void SignComputex::inplace(poplar::program::Sequence &prog,
                           poplar::Graph &graph,
                           const poplar::Tensor &tensor,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  popops::mapInPlace(
      graph, popops::expr::UnaryOpType::SIGNUM, tensor, prog, {dnai, s});
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
