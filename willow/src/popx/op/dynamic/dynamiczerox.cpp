// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Zero.hpp>
#include <popart/op/dynamic/dynamiczero.hpp>
#include <popart/popx/op/dynamic/dynamiczerox.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/dynamic/dynamicbase.hpp"
#include "popart/popx/opx.hpp"
#include "popart/region.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {

void DynamicZeroOpx::grow(poplar::program::Sequence &prog) const {
  auto &op    = getOp<DynamicBinaryBaseOp>();
  auto tensor = getInTensor(DynamicBinaryBaseOp::getUpdateInIndex());
  auto index  = getInTensor(DynamicBinaryBaseOp::getIndexInIndex());

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  auto updateShape = op.inShape(DynamicBinaryBaseOp::getUpdateInIndex());

  auto slice =
      popops::createSliceTensor(graph(), tensor, paxes, psizes, 1).squeeze({0});
  popops::zero(graph(), slice, prog, debugContext("dynamic_zero_zero"));

  auto outTensor = cloneNcopyOpt(prog, tensor);

  popops::dynamicUpdate(
      graph(),
      outTensor,
      slice,
      popops::cast(graph(),
                   index.reshape({op.getAxes().size()}),
                   poplar::UNSIGNED_INT,
                   prog,
                   debugContext()),
      paxes,
      psizes,
      prog,
      debugContext("dynamic_zero_" +
                   op.inId(DynamicBinaryBaseOp::getUpdateInIndex())));

  setOutTensor(DynamicBinaryBaseOp::getOutIndex(), outTensor);
}

poplar::Tensor DynamicZeroOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                  InIndex in,
                                                  OutIndex) const {
  if (in == DynamicZeroOp::getUpdateInIndex()) {
    return tensor;
  } else {
    return Opx::unwindTensorLayout(tensor, in, 0);
  }
}

view::RegMap DynamicZeroOpx::unwindRegion(InIndex index, OutIndex) const {
  DynamicBinaryBaseOp *op = dynamic_cast<DynamicBinaryBaseOp *>(this->op_p);
  auto shape              = op->inShape(index);
  return [shape](const view::Region &) {
    return view::Regions(1, view::Region::getFull(shape));
  };
}

poplar::Tensor DynamicZeroOpx::cloneNcopyOpt(poplar::program::Sequence &s,
                                             const poplar::Tensor &t) const {
  return cloneNcopy(s, t);
}

InputCreatorType DynamicZeroOpx::getInputCreatorType(InIndex index) const {
  return index == DynamicBinaryBaseOp::getUpdateInIndex()
             ? InputCreatorType::CanUnwind
             : Opx::getInputCreatorType(index);
}

poplar::Tensor
DynamicZeroInplaceOpx::cloneNcopyOpt(poplar::program::Sequence &s,
                                     const poplar::Tensor &t) const {
  if (t.isParallelWriteable()) {
    return t;
  } else {
    // Outplace because t has internal aliases
    return cloneNcopy(s, t);
  }
}

namespace {
// Ops
OpxCreator<DynamicZeroOpx>
    dynamicZeroOpxCreator(Onnx::CustomOperators::DynamicZero_1);
OpxCreator<DynamicZeroInplaceOpx>
    dynamicZeroInplaceOpxCreator(Onnx::CustomOperators::DynamicZeroInplace);
} // namespace

} // namespace popx
} // namespace popart
