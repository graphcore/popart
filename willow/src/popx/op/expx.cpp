// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <string>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/expx.hpp>
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

poplar::Tensor ExpComputex::outplace(poplar::program::Sequence &p,
                                     poplar::Graph &g,
                                     const poplar::Tensor &t,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &dbs) const {

  return popops::map(g, popops::expr::UnaryOpType::EXPONENT, t, p, {dnai, dbs});
}

void ExpComputex::inplace(poplar::program::Sequence &p,
                          poplar::Graph &g,
                          const poplar::Tensor &t,
                          const poplar::DebugNameAndId &dnai,
                          const std::string &dbs) const {

  popops::mapInPlace(g, popops::expr::UnaryOpType::EXPONENT, t, p, {dnai, dbs});
}

namespace {
OpxCreator<ExpOpx> expOpxCreator(Onnx::Operators::Exp_6);
OpxCreator<ExpInplaceOpx>
    expxInplaceOpxCreator(Onnx::CustomOperators::ExpInplace);
} // namespace

} // namespace popx
} // namespace popart
