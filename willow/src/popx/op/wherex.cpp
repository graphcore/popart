// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/op/where.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/wherex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace popart {
namespace popx {

WhereOpx::WhereOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<WhereOp>(op, {Onnx::Operators::Where_9});
}

void WhereOpx::grow(snap::program::Sequence &prog) const {

  const auto condition =
      getInTensor(WhereOp::conditionInIndex()).getPoplarTensor();
  const auto x = getInTensor(WhereOp::xInIndex()).getPoplarTensor();
  const auto y = getInTensor(WhereOp::yInIndex()).getPoplarTensor();

  const auto result = popops::select(graph().getPoplarGraph(),
                                     x,
                                     y,
                                     condition,
                                     prog.getPoplarSequence(),
                                     debugContext(),
                                     poplar::OptionFlags());

  setOutTensor(WhereOp::outIndex(), snap::Tensor{result, graph()});
}

WhereXGradOpx::WhereXGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<WhereXGradOp>(op, Onnx::GradOperators::WhereXGrad);
}

void WhereXGradOpx::grow(snap::program::Sequence &prog) const {

  const auto &op = getOp<WhereXGradOp>();
  const auto whereOutGrad =
      getInTensor(WhereXGradOp::outGradInIndex()).getPoplarTensor();
  const auto condition =
      getInTensor(WhereXGradOp::fwdConditionInIndex()).getPoplarTensor();

  // Copy x shape and pad to the same length as whereOutGrad
  const std::vector<size_t> xShape = op.getFwdInShape();
  std::vector<size_t> paddedXshape(xShape.begin(), xShape.end());
  while (paddedXshape.size() < whereOutGrad.shape().size()) {
    paddedXshape.insert(paddedXshape.begin(), 1);
  }

  // Find dimensions that we need to reduce in.
  std::vector<size_t> reduction_dims;
  for (size_t i = 0; i < paddedXshape.size(); i++) {
    if (paddedXshape.at(i) != whereOutGrad.shape().at(i)) {
      reduction_dims.push_back(i);
    }
  }

  auto condition2 = popops::cast(graph().getPoplarGraph(),
                                 condition,
                                 whereOutGrad.elementType(),
                                 prog.getPoplarSequence(),
                                 debugContext("cast_x"));

  auto gradX = popops::mul(graph().getPoplarGraph(),
                           whereOutGrad,
                           condition2,
                           prog.getPoplarSequence(),
                           debugContext("grad_x"));

  // Reduces the output.
  auto gradX2 = popops::reduce(graph().getPoplarGraph(),
                               gradX,
                               reduction_dims,
                               {popops::Operation::ADD},
                               prog.getPoplarSequence(),
                               debugContext("add"));
  // The reduce above will have removed all the reduction dims.
  // Some dims of size 1 may need to be added back in, we reshape.
  gradX2 = gradX2.reshape(xShape);

  setOutTensor(WhereXGradOp::outIndex(), snap::Tensor{gradX2, graph()});
}

WhereYGradOpx::WhereYGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<WhereYGradOp>(op, Onnx::GradOperators::WhereYGrad);
}

void WhereYGradOpx::grow(snap::program::Sequence &prog) const {

  const auto &op = getOp<WhereYGradOp>();
  const auto whereOutGrad =
      getInTensor(WhereYGradOp::outGradInIndex()).getPoplarTensor();
  const auto condition =
      getInTensor(WhereYGradOp::fwdConditionInIndex()).getPoplarTensor();

  const std::vector<size_t> yShape = op.getFwdInShape();

  std::vector<size_t> paddedYshape(yShape.begin(), yShape.end());
  while (paddedYshape.size() < whereOutGrad.shape().size()) {
    paddedYshape.insert(paddedYshape.begin(), 1);
  }

  std::vector<size_t> reduction_dims;
  for (size_t i = 0; i < paddedYshape.size(); i++) {
    if (paddedYshape.at(i) != whereOutGrad.shape().at(i)) {
      reduction_dims.push_back(i);
    }
  }

  poplar::Tensor condition2 = popops::logicalNot(graph().getPoplarGraph(),
                                                 condition,
                                                 prog.getPoplarSequence(),
                                                 debugContext("logical_not"));

  auto condition3 = popops::cast(graph().getPoplarGraph(),
                                 condition2,
                                 whereOutGrad.elementType(),
                                 prog.getPoplarSequence(),
                                 debugContext("cast_y"));

  auto gradY = popops::mul(graph().getPoplarGraph(),
                           whereOutGrad,
                           condition3,
                           prog.getPoplarSequence(),
                           debugContext("grad_y"));

  auto gradY2 = popops::reduce(graph().getPoplarGraph(),
                               gradY,
                               reduction_dims,
                               {popops::Operation::ADD},
                               prog.getPoplarSequence(),
                               debugContext("add"));
  gradY2      = gradY2.reshape(yShape);

  setOutTensor(WhereYGradOp::outIndex(), snap::Tensor{gradY2, graph()});
}

namespace {
OpxCreator<WhereOpx> whereOpxCreator(Onnx::Operators::Where_9);
OpxCreator<WhereXGradOpx> whereXGradOpxCreator(Onnx::GradOperators::WhereXGrad);
OpxCreator<WhereYGradOpx> whereYGradOpxCreator(Onnx::GradOperators::WhereYGrad);
} // namespace

} // namespace popx
} // namespace popart
