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

  auto inTensor = getInTensor(CacheStoreOp::getCachedTensorInIndex());

  poplar::Tensor rbTensor;

  if (!dv_p->hasRemoteBuffer(cacheStoreOp.getRemoteBufferId())) {
    rbTensor =
        graph().clone(inTensor,
                      inTensorId + "_CacheTmp",
                      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    dv_p->createRemoteBuffer(cacheStoreOp.getRemoteBufferId(), rbTensor);
  }

  auto buffer = dv_p->getRemoteBuffer(cacheStoreOp.getRemoteBufferId());

  rbTensor = buffer.second.get();

  poplar::program::Copy tmp_copy_prog(inTensor, rbTensor);
  prog.add(tmp_copy_prog);

  if (cacheStoreOp.input->hasIndex(
          CacheStoreOp::getRemoteBufferOffsetInIndex())) {
    auto offset = getInTensor(CacheStoreOp::getRemoteBufferOffsetInIndex());
    poplar::program::Copy copy_prog(rbTensor, buffer.first, offset);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(rbTensor, buffer.first);
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

  poplar::Tensor outTensor = getInTensor(CacheLoadOp::getCachedTensorInIndex());

  poplar::Tensor rbTensor;

  if (!dv_p->hasRemoteBuffer(cacheLoadOp.getRemoteBufferId())) {
    rbTensor =
        graph().clone(outTensor,
                      outTensorId + "_CacheTmp",
                      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    dv_p->createRemoteBuffer(cacheLoadOp.getRemoteBufferId(), rbTensor);
  }

  auto buffer = dv_p->getRemoteBuffer(cacheLoadOp.getRemoteBufferId());
  rbTensor    = buffer.second.get();

  if (cacheLoadOp.input->hasIndex(
          CacheLoadOp::getRemoteBufferOffsetInIndex())) {
    auto offset = getInTensor(CacheLoadOp::getRemoteBufferOffsetInIndex());
    poplar::program::Copy copy_prog(buffer.first, rbTensor, offset);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(buffer.first, rbTensor);
    prog.add(copy_prog);
  }

  poplar::program::Copy tmp_copy_prog(rbTensor, outTensor);
  prog.add(tmp_copy_prog);

  if (hasInViewChangers(CacheLoadOp::getCachedTensorInIndex())) {
    setOutViewChangers(
        CacheLoadOp::getCachedTensorOutIndex(),
        getInViewChangers(CacheLoadOp::getCachedTensorInIndex()));
  }
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
