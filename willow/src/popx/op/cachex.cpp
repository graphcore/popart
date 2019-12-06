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

  logging::trace("[CacheStoreOpx] Growing CacheStore for tensor {}, "
                 "using RemoteBuffer {}",
                 cacheStoreOp.input->tensor(0)->id,
                 cacheStoreOp.getRemoteBufferId());

  auto inTensor = getInTensor(CacheStoreOp::getCachedTensorInIndex());
  auto buffer   = dv_p->getRemoteBuffer(cacheStoreOp.getRemoteBufferId());

  if (cacheStoreOp.input->hasIndex(
          CacheStoreOp::getRemoteBufferOffsetInIndex())) {
    auto offset = getInTensor(CacheStoreOp::getRemoteBufferOffsetInIndex());
    poplar::program::Copy copy_prog(inTensor, buffer, offset);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(inTensor, buffer);
    prog.add(copy_prog);
  }
}

CacheAllocateOpx::CacheAllocateOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<CacheAllocateOp>(op, Onnx::CustomOperators::CacheAllocate);
}

void CacheAllocateOpx::grow(poplar::program::Sequence &) const {}

CacheLoadOpx::CacheLoadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CacheLoadOp>(op, Onnx::CustomOperators::CacheLoad);
}

void CacheLoadOpx::grow(poplar::program::Sequence &prog) const {
  auto &cacheLoadOp = getOp<CacheLoadOp>();

  // Tensor completely overwritten
  logging::trace(
      "[CacheLoadOpx] Growing CacheLoad for tensor {} -> {}, "
      "using RemoteBuffer {}",
      cacheLoadOp.input->tensor(CacheLoadOp::getCachedTensorInIndex())->id,
      cacheLoadOp.output->tensor(CacheLoadOp::getCachedTensorOutIndex())->id,
      cacheLoadOp.getRemoteBufferId());

  auto buffer = dv_p->getRemoteBuffer(cacheLoadOp.getRemoteBufferId());
  poplar::Tensor outTensor = getInTensor(CacheLoadOp::getCachedTensorInIndex());

  if (cacheLoadOp.input->hasIndex(
          CacheLoadOp::getRemoteBufferOffsetInIndex())) {
    auto offset = getInTensor(CacheLoadOp::getRemoteBufferOffsetInIndex());
    poplar::program::Copy copy_prog(buffer, outTensor, offset);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(buffer, outTensor);
    prog.add(copy_prog);
  }

  setOutTensor(CacheLoadOp::getCachedTensorOutIndex(), outTensor);
}

InputCreatorType CacheLoadOpx::getInputCreatorType(InIndex index) const {
  return index == CacheLoadOp::getCachedTensorInIndex()
             ? InputCreatorType::CANUNWIND
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
OpxCreator<CacheAllocateOpx>
    cacheAllocateOpxCreator(Onnx::CustomOperators::CacheAllocate);
OpxCreator<CacheStoreOpx>
    cacheStoreOpxCreator(Onnx::CustomOperators::CacheStore);
OpxCreator<CacheLoadOpx> cacheLoadOpxCreator(Onnx::CustomOperators::CacheLoad);
} // namespace
} // namespace popx
} // namespace popart
