// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/cache.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/cachex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

CacheStoreOpx::CacheStoreOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CacheStoreOp>(op, Onnx::CustomOperators::CacheStore);
}

void CacheStoreOpx::grow(poplar::program::Sequence &prog) const {
  auto &cacheStoreOp = getOp<CacheStoreOp>();

  TensorId inTensorId =
      cacheStoreOp.input->tensor(CacheStoreOp::getCachedTensorInIndex())->id;

  logging::debug(
      "[CacheStoreOpx] Growing CacheStore for tensor {}, "
      "using RemoteBuffer {}",
      cacheStoreOp.input->tensor(CacheStoreOp::getCachedTensorInIndex())->id,
      cacheStoreOp.getRemoteBufferId());

  auto buffer   = dv_p->getRemoteBuffer(cacheStoreOp.getRemoteBufferId());
  auto inTensor = getInTensor(CacheStoreOp::getCachedTensorInIndex());

  auto rbTensor = buffer.second;

  if (!rbTensor.is_initialized()) {
    rbTensor =
        graph().clone(inTensor,
                      inTensorId + "_CacheLoadTmp",
                      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    dv_p->setRemoteBufferTensor(cacheStoreOp.getRemoteBufferId(),
                                rbTensor.get());
  }

  poplar::program::Copy tmp_copy_prog(inTensor, rbTensor.get());
  prog.add(tmp_copy_prog);

  if (cacheStoreOp.input->hasIndex(
          CacheStoreOp::getRemoteBufferOffsetInIndex())) {
    auto offset = getInTensor(CacheStoreOp::getRemoteBufferOffsetInIndex());
    poplar::program::Copy copy_prog(rbTensor.get(), buffer.first, offset);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(rbTensor.get(), buffer.first);
    prog.add(copy_prog);
  }
}

CacheLoadOpx::CacheLoadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CacheLoadOp>(op, Onnx::CustomOperators::CacheLoad);
}

void CacheLoadOpx::grow(poplar::program::Sequence &prog) const {
  auto &cacheLoadOp = getOp<CacheLoadOp>();

  TensorId outTensorId =
      cacheLoadOp.output->tensor(CacheLoadOp::getCachedTensorOutIndex())->id;

  // Tensor completely overwritten
  logging::debug(
      "[CacheLoadOpx] Growing CacheLoad for tensor {} -> {}, "
      "using RemoteBuffer {}",
      cacheLoadOp.input->tensor(CacheLoadOp::getCachedTensorInIndex())->id,
      cacheLoadOp.output->tensor(CacheLoadOp::getCachedTensorOutIndex())->id,
      cacheLoadOp.getRemoteBufferId());

  auto buffer = dv_p->getRemoteBuffer(cacheLoadOp.getRemoteBufferId());
  poplar::Tensor outTensor = getInTensor(CacheLoadOp::getCachedTensorInIndex());

  auto rbTensor = buffer.second;

  if (!rbTensor.is_initialized()) {
    rbTensor =
        graph().clone(outTensor,
                      outTensorId + "_CacheLoadTmp",
                      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    dv_p->setRemoteBufferTensor(cacheLoadOp.getRemoteBufferId(),
                                rbTensor.get());
  }

  if (cacheLoadOp.input->hasIndex(
          CacheLoadOp::getRemoteBufferOffsetInIndex())) {
    auto offset = getInTensor(CacheLoadOp::getRemoteBufferOffsetInIndex());
    poplar::program::Copy copy_prog(buffer.first, rbTensor.get(), offset);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(buffer.first, rbTensor.get());
    prog.add(copy_prog);
  }

  poplar::program::Copy tmp_copy_prog(rbTensor.get(), outTensor);
  prog.add(tmp_copy_prog);

  setOutTensor(CacheLoadOp::getCachedTensorOutIndex(), outTensor);
}

InputCreatorType CacheLoadOpx::getInputCreatorType(InIndex index) const {
  return index == CacheLoadOp::getCachedTensorInIndex()
             ? InputCreatorType::CanUnwind
             : Opx::getInputCreatorType(index);
}

poplar::Tensor CacheLoadOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                InIndex,
                                                OutIndex) const {
  return tensor;
}

view::RegMap CacheLoadOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

namespace {
OpxCreator<CacheStoreOpx>
    cacheStoreOpxCreator(Onnx::CustomOperators::CacheStore);
OpxCreator<CacheLoadOpx> cacheLoadOpxCreator(Onnx::CustomOperators::CacheLoad);
} // namespace
} // namespace popx
} // namespace popart
