// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/remote.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/remotex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

RemoteStoreOpx::RemoteStoreOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<RemoteStoreOp>(op, Onnx::CustomOperators::RemoteStore);
}

void RemoteStoreOpx::grow(poplar::program::Sequence &prog) const {
  auto &remoteStoreOp = getOp<RemoteStoreOp>();

  TensorId inTensorId =
      remoteStoreOp.input->tensor(RemoteStoreOp::getLocalTensorInIndex())->id;

  logging::debug(
      "[RemoteStoreOpx] Growing RemoteStore for tensor {}, "
      "using RemoteBuffer {}",
      remoteStoreOp.input->tensor(RemoteStoreOp::getLocalTensorInIndex())->id,
      remoteStoreOp.getRemoteBufferId());

  auto inTensor = getInTensor(RemoteStoreOp::getLocalTensorInIndex());

  poplar::Tensor rbTensor;

  if (!dv_p->hasRemoteBuffer(remoteStoreOp.getRemoteBufferId())) {
    rbTensor =
        graph().clone(inTensor,
                      inTensorId + "_RemoteTmp",
                      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    dv_p->createRemoteBuffer(remoteStoreOp.getRemoteBufferId(), rbTensor);
  }

  auto buffer = dv_p->getRemoteBuffer(remoteStoreOp.getRemoteBufferId());

  rbTensor = buffer.second.value();

  poplar::program::Copy tmp_copy_prog(inTensor, rbTensor);
  prog.add(tmp_copy_prog);

  if (remoteStoreOp.input->hasIndex(
          RemoteStoreOp::getRemoteBufferOffsetInIndex())) {
    auto offset = getInTensor(RemoteStoreOp::getRemoteBufferOffsetInIndex());
    poplar::program::Copy copy_prog(rbTensor, buffer.first, offset);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(rbTensor, buffer.first);
    prog.add(copy_prog);
  }
}

RemoteLoadOpx::RemoteLoadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<RemoteLoadOp>(op, Onnx::CustomOperators::RemoteLoad);
}

void RemoteLoadOpx::grow(poplar::program::Sequence &prog) const {
  auto &remoteLoadOp = getOp<RemoteLoadOp>();

  TensorId outTensorId =
      remoteLoadOp.output->tensor(RemoteLoadOp::getLocalTensorOutIndex())->id;

  // Tensor completely overwritten
  logging::debug(
      "[RemoteLoadOpx] Growing RemoteLoad for tensor {} -> {}, "
      "using RemoteBuffer {}",
      remoteLoadOp.input->tensor(RemoteLoadOp::getLocalTensorInIndex())->id,
      remoteLoadOp.output->tensor(RemoteLoadOp::getLocalTensorOutIndex())->id,
      remoteLoadOp.getRemoteBufferId());

  poplar::Tensor outTensor = getInTensor(RemoteLoadOp::getLocalTensorInIndex());

  poplar::Tensor rbTensor;

  if (!dv_p->hasRemoteBuffer(remoteLoadOp.getRemoteBufferId())) {
    rbTensor =
        graph().clone(outTensor,
                      outTensorId + "_RemoteTmp",
                      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    dv_p->createRemoteBuffer(remoteLoadOp.getRemoteBufferId(), rbTensor);
  }

  auto buffer = dv_p->getRemoteBuffer(remoteLoadOp.getRemoteBufferId());
  rbTensor    = buffer.second.value();

  if (remoteLoadOp.input->hasIndex(
          RemoteLoadOp::getRemoteBufferOffsetInIndex())) {
    auto offset = getInTensor(RemoteLoadOp::getRemoteBufferOffsetInIndex());
    poplar::program::Copy copy_prog(buffer.first, rbTensor, offset);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(buffer.first, rbTensor);
    prog.add(copy_prog);
  }

  poplar::program::Copy tmp_copy_prog(rbTensor, outTensor);
  prog.add(tmp_copy_prog);

  if (hasInViewChangers(RemoteLoadOp::getLocalTensorInIndex())) {
    setOutViewChangers(
        RemoteLoadOp::getLocalTensorOutIndex(),
        getInViewChangers(RemoteLoadOp::getLocalTensorInIndex()));
  }
  setOutTensor(RemoteLoadOp::getLocalTensorOutIndex(), outTensor);
}

InputCreatorType RemoteLoadOpx::getInputCreatorType(InIndex index) const {
  return index == RemoteLoadOp::getLocalTensorInIndex()
             ? InputCreatorType::CanUnwind
             : Opx::getInputCreatorType(index);
}

poplar::Tensor RemoteLoadOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                 InIndex,
                                                 OutIndex) const {
  return tensor;
}

view::RegMap RemoteLoadOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

namespace {
OpxCreator<RemoteStoreOpx>
    remoteStoreOpxCreator(Onnx::CustomOperators::RemoteStore);
OpxCreator<RemoteLoadOpx>
    remoteLoadOpxCreator(Onnx::CustomOperators::RemoteLoad);
} // namespace
} // namespace popx
} // namespace popart
