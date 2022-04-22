// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/expm1x.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class Expm1InplaceOp;
class Expm1Op;
class Op;

namespace popx {
class Devicex;

Expm1InplaceOpx::Expm1InplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, Expm1Computex::get()) {
  verifyOp<Expm1InplaceOp>(op, Onnx::CustomOperators::Expm1Inplace);
}

Expm1Opx::Expm1Opx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, Expm1Computex::get()) {
  verifyOp<Expm1Op>(op, Onnx::CustomOperators::Expm1_1);
}

snap::Tensor Expm1Computex::outplace(snap::program::Sequence &p,
                                     snap::Graph &g,
                                     const snap::Tensor &t,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &dbs) const {

  return snap::Tensor{popops::map(g.getPoplarGraph(),
                                  popops::expr::UnaryOpType::EXPONENT_MINUS_ONE,
                                  t.getPoplarTensor(),
                                  p.getPoplarSequence(),
                                  {dnai, dbs}),
                      g};
}

void Expm1Computex::inplace(snap::program::Sequence &p,
                            snap::Graph &g,
                            const snap::Tensor &t,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &dbs) const {

  popops::mapInPlace(g.getPoplarGraph(),
                     popops::expr::UnaryOpType::EXPONENT_MINUS_ONE,
                     t.getPoplarTensor(),
                     p.getPoplarSequence(),
                     {dnai, dbs});
}

namespace {
OpxCreator<Expm1Opx> expm1OpxCreator(Onnx::CustomOperators::Expm1_1);
OpxCreator<Expm1InplaceOpx>
    expm1xInplaceOpxCreator(Onnx::CustomOperators::Expm1Inplace);
} // namespace

} // namespace popx
} // namespace popart
