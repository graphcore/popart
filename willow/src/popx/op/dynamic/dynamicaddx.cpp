// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/dynamic/dynamicadd.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/dynamic/dynamicaddx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <snap/popops/ElementWise.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

void DynamicAddOpx::grow(snap::program::Sequence &prog) const {
  auto &op    = getOp<DynamicTernaryBaseOp>();
  auto tensor = getInTensor(DynamicTernaryBaseOp::getUpdateInIndex());
  auto index =
      getInTensor(DynamicTernaryBaseOp::getIndexInIndex()).getPoplarTensor();
  auto slice = getInTensor(DynamicTernaryBaseOp::getInIndex());

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  auto outTensor = cloneNcopyOpt(prog, tensor);

  // Get the slice that is to be added to: s = t[index:index+psizes]
  auto s = snap::Tensor{
      popops::dynamicSlice(
          graph().getPoplarGraph(),
          tensor.getPoplarTensor(),
          popops::cast(graph().getPoplarGraph(),
                       index.reshape({op.getAxes().size()}),
                       poplar::UNSIGNED_INT,
                       prog.getPoplarSequence(),
                       debugContext()),
          paxes,
          psizes,
          prog.getPoplarSequence(),
          debugContext("dynamic_add_slice_" +
                       op.inId(DynamicTernaryBaseOp::getUpdateInIndex()))),
      graph()};

  // Add inplace: s += slice
  snap::popops::mapInPlace(
      graph(),
      popops::expr::BinaryOpType::ADD,
      s,
      slice,
      prog,
      debugContext("dynamic_add_mip_" +
                   op.inId(DynamicTernaryBaseOp::getUpdateInIndex())));

  // Update: t[index:index+psizes] = s
  popops::dynamicUpdate(
      graph().getPoplarGraph(),
      outTensor.getPoplarTensor(),
      s.getPoplarTensor(),
      popops::cast(graph().getPoplarGraph(),
                   index.reshape({op.getAxes().size()}),
                   poplar::UNSIGNED_INT,
                   prog.getPoplarSequence(),
                   debugContext()),
      paxes,
      psizes,
      prog.getPoplarSequence(),
      debugContext("dynamic_add_" +
                   op.inId(DynamicTernaryBaseOp::getUpdateInIndex())));

  setOutTensor(DynamicTernaryBaseOp::getOutIndex(), outTensor);
}

snap::Tensor DynamicAddInplaceOpx::cloneNcopyOpt(snap::program::Sequence &s,
                                                 const snap::Tensor &t) const {
  if (t.isParallelWriteable()) {
    return t;
  } else {
    // Outplace because t has internal aliases
    return cloneNcopy(s, t);
  }
}

namespace {
// Ops
OpxCreator<DynamicAddOpx>
    dynamicAddOpxCreator(Onnx::CustomOperators::DynamicAdd_1);
OpxCreator<DynamicAddInplaceOpx>
    dynamicAddInplaceOpxCreator(Onnx::CustomOperators::DynamicAddInplace);

} // namespace

} // namespace popx
} // namespace popart
