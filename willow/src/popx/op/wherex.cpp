// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popart/op/where.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/wherex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"

namespace pe = popops::expr;

namespace popart {

namespace popx {

BaseWhereOpx::BaseWhereOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

void BaseWhereOpx::grow(snap::program::Sequence &prog) const {
  const auto condition = getInTensor(WhereOp::conditionInIndex());
  auto x               = getInTensor(WhereOp::xInIndex());
  auto y               = getInTensor(WhereOp::yInIndex());

  doGrow(prog, x, y, condition);
}

InputCreatorType BaseWhereOpx::getInputCreatorType(InIndex inIndex) const {
  // If it's broadcasted, revert to linear
  if (op_p->inInfo(inIndex).shape() !=
      op_p->outInfo(WhereOp::outIndex()).shape()) {
    return InputCreatorType::Deadend;
  }
  // To avoid complications, always unwind on one index
  // Favour the order x, y, condition
  InIndex unwindIdx = unwindIndex();
  if (inIndex == unwindIdx) {
    // TODO T59196: Set to CanUnwind
    return InputCreatorType::Deadend;
  }

  // On other indicies, allow a creator which clones the input matching
  // unwindIndex. If we got here, at least this input and the unwindIndex match
  // the output size.
  // TODO T59196: Set to CanCreateOrUnwind
  return InputCreatorType::Deadend;
}

snap::Tensor BaseWhereOpx::unwindTensorLayout(snap::Tensor tensor,
                                              InIndex inIndex,
                                              OutIndex) const {
  if (inIndex == WhereOp::conditionInIndex()) {
    return graph().clone(
        popType(op_p->inInfo(inIndex)), tensor, debugContext());
  }

  return tensor;
}

std::set<TensorId> BaseWhereOpx::mustExistBeforeCreate(InIndex) const {
  std::set<TensorId> mustExist;
  mustExist.insert(op_p->input->tensor(unwindIndex())->id);
  return mustExist;
}

snap::Tensor
BaseWhereOpx::createInputTensor(InIndex inIndex,
                                const poplar::DebugNameAndId &dnai) const {
  const auto unwindIdx = unwindIndex();
  if (!dv_p->lowering().tensors().contains(op_p->input->id(unwindIdx))) {
    throw internal_error("WhereOpx::createInputTensor invalid configuration.");
  }

  return graph().clone(
      popType(op_p->inInfo(inIndex)), getInTensor(unwindIdx), dnai);
}

view::RegMap BaseWhereOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

void BaseWhereOpx::outplace(snap::program::Sequence &prog,
                            const snap::Tensor &x,
                            const snap::Tensor &y,
                            const snap::Tensor &condition) const {
  const auto result = popops::select(graph().getPoplarGraph(),
                                     x.getPoplarTensor(),
                                     y.getPoplarTensor(),
                                     condition.getPoplarTensor(),
                                     prog.getPoplarSequence(),
                                     debugContext());
  setOutTensor(WhereOp::outIndex(), snap::Tensor{result, graph()});
}

InIndex BaseWhereOpx::unwindIndex() const {
  if (op_p->inShape(WhereOp::xInIndex()) ==
      op_p->outShape(WhereOp::outIndex())) {
    return WhereOp::xInIndex();
  }

  if (op_p->inShape(WhereOp::yInIndex()) ==
      op_p->outShape(WhereOp::outIndex())) {
    return WhereOp::yInIndex();
  }

  if (op_p->inShape(WhereOp::conditionInIndex()) ==
      op_p->outShape(WhereOp::outIndex())) {
    return WhereOp::conditionInIndex();
  }

  // Technically, all three may broadcast different dimensions.
  return -1;
}

WhereOpx::WhereOpx(Op *op, Devicex *devicex) : BaseWhereOpx(op, devicex) {
  verifyOp<WhereOp>(op, {Onnx::Operators::Where_9});
}

void WhereOpx::doGrow(snap::program::Sequence &prog,
                      const snap::Tensor &x,
                      const snap::Tensor &y,
                      const snap::Tensor &condition) const {
  outplace(prog, x, y, condition);
}

WhereLhsInplaceOpx::WhereLhsInplaceOpx(Op *op, Devicex *devicex)
    : BaseWhereOpx(op, devicex) {
  verifyOp<WhereLhsInplaceOp>(op);
}

void WhereLhsInplaceOpx::doGrow(snap::program::Sequence &prog,
                                const snap::Tensor &x,
                                const snap::Tensor &y,
                                const snap::Tensor &condition) const {
  // Fallback to outplace if not parallel writable
  if (!x.isParallelWriteable()) {
    outplace(prog, x, y, condition);
    return;
  }

  popops::selectInPlace(graph().getPoplarGraph(),
                        x.getPoplarTensor(),
                        y.getPoplarTensor(),
                        condition.getPoplarTensor(),
                        prog.getPoplarSequence(),
                        debugContext());

  setOutTensor(WhereOp::outIndex(), x);
}

WhereRhsInplaceOpx::WhereRhsInplaceOpx(Op *op, Devicex *devicex)
    : BaseWhereOpx(op, devicex) {
  verifyOp<WhereRhsInplaceOp>(op);
}

void WhereRhsInplaceOpx::doGrow(snap::program::Sequence &prog,
                                const snap::Tensor &x,
                                const snap::Tensor &y,
                                const snap::Tensor &condition) const {
  // Fallback to outplace if not parallel writable
  if (!y.isParallelWriteable()) {
    outplace(prog, x, y, condition);
    return;
  }

  // Reverse the order and use not to reverse condition
  auto expr = pe::Select(pe::_1, pe::_2, pe::Not(pe::_3));

  snap::popops::mapInPlace(
      graph(), expr, {y, x, condition}, prog, debugContext());

  setOutTensor(WhereOp::outIndex(), y);
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
OpxCreator<WhereLhsInplaceOpx>
    whereLhsInplaceOpxCreator(Onnx::CustomOperators::WhereLhsInplace);
OpxCreator<WhereRhsInplaceOpx>
    whereRhsInplaceOpxCreator(Onnx::CustomOperators::WhereRhsInplace);
OpxCreator<WhereXGradOpx> whereXGradOpxCreator(Onnx::GradOperators::WhereXGrad);
OpxCreator<WhereYGradOpx> whereYGradOpxCreator(Onnx::GradOperators::WhereYGrad);
} // namespace

} // namespace popx
} // namespace popart
