// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/lrn.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/lrnx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace popart {
namespace popx {

LRNOpx::LRNOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<LRNOp>(op, {Onnx::Operators::LRN_1});
}

namespace {
poplar::Tensor getScale(snap::Graph &graph,
                        const poplar::Tensor &input,
                        snap::program::Sequence &prog,
                        const float alpha,
                        const float bias,
                        const int64_t size,
                        const poplar::DebugContext &debugContext) {
  const poplar::DebugInfo di(debugContext, "");
  auto square = popops::square(
      graph.getPoplarGraph(), input, prog.getPoplarSequence(), {di});
  auto square_sum = graph.getPoplarGraph().clone(square, {di});
  prog.add(poplar::program::Copy(square, square_sum, false, {di}));
  auto channels = input.dim(1);

  auto left  = ((size - 1) / 2);
  auto right = size - left;

  for (auto i = -left; i < right; ++i) {
    // i == 0 added by default,
    if ((i != 0L) &&
        (channels - std::max<int64_t>(0L, i)) - std::max<int64_t>(0L, -i) > 0)
      popops::addInPlace(graph.getPoplarGraph(),
                         square_sum.slice(std::max<int64_t>(0L, -i),
                                          channels - std::max<int64_t>(0L, i),
                                          1),
                         square.slice(std::max<int64_t>(0L, i),
                                      channels - std::max<int64_t>(0L, -i),
                                      1),
                         prog.getPoplarSequence(),
                         {di});
  }

  auto scale = popops::map(
      graph.getPoplarGraph(),
      pe::Add(pe::Const(bias), pe::Mul(pe::Const(alpha / size), pe::_1)),
      {square_sum},
      prog.getPoplarSequence(),
      {di});

  return scale;
}
} // namespace

void LRNOpx::grow(snap::program::Sequence &prog) const {
  const auto &op   = getOp<LRNOp>();
  const auto input = getInTensor(LRNOp::getInIndex()).getPoplarTensor();

  auto scale = getScale(graph(),
                        input,
                        prog,
                        op.getAlpha(),
                        op.getBias(),
                        op.getSize(),
                        debugContext("scale"));

  auto output =
      popops::map(graph().getPoplarGraph(),
                  pe::Mul(pe::_1, pe::Pow(pe::_2, pe::Const(-op.getBeta()))),
                  {input, scale},
                  prog.getPoplarSequence(),
                  debugContext("output"));

  setOutTensor(LRNOp::getOutIndex(), snap::Tensor{output, graph()});
}

LRNGradOpx::LRNGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<LRNGradOp>(op, Onnx::GradOperators::LRNGrad);
}

void LRNGradOpx::grow(snap::program::Sequence &prog) const {
  const auto &op   = getOp<LRNGradOp>();
  const auto input = getInTensor(LRNGradOp::getInIndex()).getPoplarTensor();
  const auto fwd_input =
      getInTensor(LRNGradOp::getFwdInInIndex()).getPoplarTensor();

  auto scale = getScale(graph(),
                        fwd_input,
                        prog,
                        op.getAlpha(),
                        op.getBias(),
                        op.getSize(),
                        debugContext("scale"));

  auto output = popops::map(
      graph().getPoplarGraph(),
      pe::Mul(
          pe::_1,
          pe::Sub(
              pe::Pow(pe::_3, pe::Const(-op.getBeta())),
              pe::Mul(pe::Mul(pe::Square(pe::_2),
                              pe::Const(2.f * op.getAlpha() * op.getBeta())),
                      pe::Pow(pe::_3, pe::Const(-op.getBeta() - 1.f))))),
      {input, fwd_input, scale},
      prog.getPoplarSequence(),
      debugContext("grad"));

  setOutTensor(LRNGradOp::getOutIndex(), snap::Tensor{output, graph()});
}

namespace {
OpxCreator<LRNOpx> batchNormOpxCreator({Onnx::Operators::LRN_1});
OpxCreator<LRNGradOpx> batchNormGradOpxCreator(Onnx::GradOperators::LRNGrad);
} // namespace

} // namespace popx
} // namespace popart
