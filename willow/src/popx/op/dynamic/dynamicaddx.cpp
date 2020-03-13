// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/dynamic/dynamicadd.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/dynamic/dynamicaddx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

void DynamicAddOpx::grow(poplar::program::Sequence &prog) const {
  auto &op    = getOp<DynamicTernaryBaseOp>();
  auto tensor = getInTensor(DynamicTernaryBaseOp::getUpdateInIndex());
  auto index  = getInTensor(DynamicTernaryBaseOp::getIndexInIndex());
  auto slice  = getInTensor(DynamicTernaryBaseOp::getInIndex());

  std::vector<size_t> paxes(op.getAxes().begin(), op.getAxes().end());
  std::vector<size_t> psizes(op.getSizes().begin(), op.getSizes().end());

  auto outTensor = cloneNcopyOpt(prog, tensor);

  // Get the slice that is to be added to: s = t[index:index+psizes]
  auto s = popops::dynamicSlice(
      graph(),
      tensor,
      index,
      paxes,
      psizes,
      prog,
      debugPrefix("dynamic_add_slice_" +
                  op.inId(DynamicTernaryBaseOp::getUpdateInIndex())));

  // Add inplace: s += slice
  popops::mapInPlace(
      graph(),
      popops::expr::BinaryOpType::ADD,
      s,
      slice,
      prog,
      debugPrefix("dynamic_add_mip_" +
                  op.inId(DynamicTernaryBaseOp::getUpdateInIndex())));

  // Update: t[index:index+psizes] = s
  popops::dynamicUpdate(
      graph(),
      outTensor,
      s,
      index,
      paxes,
      psizes,
      prog,
      debugPrefix("dynamic_add_" +
                  op.inId(DynamicTernaryBaseOp::getUpdateInIndex())));

  setOutTensor(DynamicTernaryBaseOp::getOutIndex(), outTensor);
}

poplar::Tensor
DynamicAddInplaceOpx::cloneNcopyOpt(poplar::program::Sequence &,
                                    const poplar::Tensor &t) const {
  return t;
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
