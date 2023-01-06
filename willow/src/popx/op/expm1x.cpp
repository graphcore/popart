// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <string>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/expm1x.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace poplar {
class Graph;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

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

poplar::Tensor Expm1Computex::outplace(poplar::program::Sequence &p,
                                       poplar::Graph &g,
                                       const poplar::Tensor &t,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &dbs) const {

  return popops::map(
      g, popops::expr::UnaryOpType::EXPONENT_MINUS_ONE, t, p, {dnai, dbs});
}

void Expm1Computex::inplace(poplar::program::Sequence &p,
                            poplar::Graph &g,
                            const poplar::Tensor &t,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &dbs) const {

  popops::mapInPlace(
      g, popops::expr::UnaryOpType::EXPONENT_MINUS_ONE, t, p, {dnai, dbs});
}

namespace {
OpxCreator<Expm1Opx> expm1OpxCreator(Onnx::CustomOperators::Expm1_1);
OpxCreator<Expm1InplaceOpx>
    expm1xInplaceOpxCreator(Onnx::CustomOperators::Expm1Inplace);
} // namespace

} // namespace popx
} // namespace popart
