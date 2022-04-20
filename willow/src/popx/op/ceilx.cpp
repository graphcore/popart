// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <vector>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/ceilx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class CeilInplaceOp;
class CeilOp;
class Op;

namespace popx {
class Devicex;

snap::Tensor CeilComputex::outplace(snap::program::Sequence &prog,
                                    snap::Graph &graph,
                                    const snap::Tensor &tensor,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &s) const {

  return snap::Tensor{popops::map(graph.getPoplarGraph(),
                                  popops::expr::UnaryOpType::CEIL,
                                  tensor.getPoplarTensor(),
                                  prog.getPoplarSequence(),
                                  {dnai, s}),
                      graph};
}

void CeilComputex::inplace(snap::program::Sequence &prog,
                           snap::Graph &graph,
                           const snap::Tensor &tensor,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  popops::mapInPlace(graph.getPoplarGraph(),
                     popops::expr::UnaryOpType::CEIL,
                     tensor.getPoplarTensor(),
                     prog.getPoplarSequence(),
                     {dnai, s});
}

CeilOpx::CeilOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, CeilComputex::get()) {
  verifyOp<CeilOp>(op, {Onnx::Operators::Ceil_1, Onnx::Operators::Ceil_6});
}

CeilInplaceOpx::CeilInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, CeilComputex::get()) {
  verifyOp<CeilInplaceOp>(op, Onnx::CustomOperators::CeilInplace);
}

namespace {
OpxCreator<CeilOpx> ceilOpxCreator({Onnx::Operators::Ceil_1,
                                    Onnx::Operators::Ceil_6});
OpxCreator<CeilInplaceOpx>
    ceilxInplaceOpxCreator(Onnx::CustomOperators::CeilInplace);
} // namespace

} // namespace popx
} // namespace popart
