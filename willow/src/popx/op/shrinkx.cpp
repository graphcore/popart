// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/error.hpp>
#include <popart/op/shrink.hpp>
#include <popart/popx/op/shrinkx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
namespace popx {
class Devicex;

namespace {
template <typename T> T *get_as(Op *op) {
  auto x = dynamic_cast<T *>(op);
  if (!x) {
    throw error("Failed to cast {} in Shrink", op->str());
  }
  return x;
}
} // namespace

ShrinkOpx::ShrinkOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          ShrinkComputex::get(get_as<ShrinkOp>(op)->lambd(),
                              get_as<ShrinkOp>(op)->bias())) {
  verifyOp<ShrinkOp>(op, {Onnx::Operators::Shrink_9});
}

poplar::Tensor ShrinkComputex::outplace(poplar::program::Sequence &prog,
                                        poplar::Graph &graph,
                                        const poplar::Tensor &tensor,
                                        const poplar::DebugNameAndId &dnai,
                                        const std::string &debug_prefix) const {
  auto out_tensor = cloneNcopy(prog, graph, tensor, dnai);
  inplace(prog, graph, out_tensor, dnai, debug_prefix);
  return out_tensor;
}

void ShrinkComputex::inplace(poplar::program::Sequence &prog,
                             poplar::Graph &graph,
                             const poplar::Tensor &tensor,
                             const poplar::DebugNameAndId &dnai,
                             const std::string &debug_prefix) const {
  popops::mapInPlace(
      graph,
      pe::Select(pe::Add(pe::_1, pe::Const(this->bias())),
                 pe::Select(pe::Sub(pe::_1, pe::Const(this->bias())),
                            pe::Const(0.0f),
                            pe::Gt(pe::_1, pe::Const(this->lambd()))),
                 pe::Lt(pe::_1, pe::Const(-this->lambd()))),
      {tensor},
      prog,
      {dnai, debug_prefix});
}

ShrinkInplaceOpx::ShrinkInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          ShrinkComputex::get(get_as<ShrinkInplaceOp>(op)->lambd(),
                              get_as<ShrinkInplaceOp>(op)->bias())) {
  verifyOp<ShrinkInplaceOp>(op, Onnx::CustomOperators::ShrinkInplace);
}

ShrinkGradOpx::ShrinkGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ShrinkGradOp>(op, Onnx::GradOperators::ShrinkGrad);
}

void ShrinkGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto &op       = getOp<ShrinkGradOp>();
  const auto input     = getInTensor(ShrinkGradOp::getGradInIndex());
  const auto fwd_input = getInTensor(ShrinkGradOp::getFwdArgInIndex());

  std::vector<std::unique_ptr<popops::expr::Expr>> exprs;
  exprs.push_back(
      std::make_unique<pe::Sub>(pe::Abs(pe::_2), pe::Const(op.lambd())));
  exprs.push_back(std::make_unique<pe::Signum>(*exprs.back()));
  exprs.push_back(std::make_unique<pe::Add>(pe::Const(1.0f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::Const(0.5f), *exprs.back()));
  exprs.push_back(std::make_unique<pe::Mul>(pe::_1, *exprs.back()));

  auto output = popops::map(graph(),
                            *exprs.back(),
                            {input, fwd_input},
                            prog,
                            debugContext("output_grad"));

  setOutTensor(ShrinkGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ShrinkOpx> shrinkOpxCreator(Onnx::Operators::Shrink_9);
OpxCreator<ShrinkInplaceOpx>
    shrinkInplaceOpxCreator(Onnx::CustomOperators::ShrinkInplace);
OpxCreator<ShrinkGradOpx> shrinkGradOpxCreator(Onnx::GradOperators::ShrinkGrad);
} // namespace

} // namespace popx
} // namespace popart
