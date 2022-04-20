// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/expx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class ExpInplaceOp;
class ExpOp;
class Op;

namespace popx {
class Devicex;

ExpInplaceOpx::ExpInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, ExpComputex::get()) {
  verifyOp<ExpInplaceOp>(op, Onnx::CustomOperators::ExpInplace);
}

ExpOpx::ExpOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, ExpComputex::get()) {
  verifyOp<ExpOp>(op, Onnx::Operators::Exp_6);
}

snap::Tensor ExpComputex::outplace(snap::program::Sequence &p,
                                   snap::Graph &g,
                                   const snap::Tensor &t,
                                   const poplar::DebugNameAndId &dnai,
                                   const std::string &dbs) const {

  return snap::Tensor{popops::map(g.getPoplarGraph(),
                                  popops::expr::UnaryOpType::EXPONENT,
                                  t.getPoplarTensor(),
                                  p.getPoplarSequence(),
                                  {dnai, dbs}),
                      g};
}

void ExpComputex::inplace(snap::program::Sequence &p,
                          snap::Graph &g,
                          const snap::Tensor &t,
                          const poplar::DebugNameAndId &dnai,
                          const std::string &dbs) const {

  snap::popops::mapInPlace(
      g, popops::expr::UnaryOpType::EXPONENT, t, p, {dnai, dbs});
}

namespace {
OpxCreator<ExpOpx> expOpxCreator(Onnx::Operators::Exp_6);
OpxCreator<ExpInplaceOpx>
    expxInplaceOpxCreator(Onnx::CustomOperators::ExpInplace);
} // namespace

} // namespace popx
} // namespace popart
