// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/signx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class Op;
class SignInplaceOp;
class SignOp;

namespace popx {
class Devicex;

snap::Tensor SignComputex::outplace(snap::program::Sequence &prog,
                                    snap::Graph &graph,
                                    const snap::Tensor &tensor,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &s) const {

  return snap::Tensor{popops::map(graph.getPoplarGraph(),
                                  popops::expr::UnaryOpType::SIGNUM,
                                  tensor.getPoplarTensor(),
                                  prog.getPoplarSequence(),
                                  {dnai, s}),
                      graph};
}

void SignComputex::inplace(snap::program::Sequence &prog,
                           snap::Graph &graph,
                           const snap::Tensor &tensor,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  snap::popops::mapInPlace(
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
