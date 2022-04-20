// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <snap/Tensor.hpp>
#include <string>
#include <utility>
#include <vector>
#include <popart/op/exchange/remote.hpp>
#include <popart/popx/op/exchange/remotex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/op/exchange/exchangex.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

// RemoteBaseOpx
RemoteBaseOpx::RemoteBaseOpx(Op *op, Devicex *devicex)
    : ExchangeBaseOpx(op, devicex) {}

RemoteStoreOpx::RemoteStoreOpx(Op *op, Devicex *devicex)
    : RemoteBaseOpx(op, devicex) {
  verifyOp<RemoteStoreOp>(op, Onnx::CustomOperators::RemoteStore);
}

// RemoteStoreOpx
void RemoteStoreOpx::grow(snap::program::Sequence &prog) const {
  auto &remoteStoreOp = getOp<RemoteStoreOp>();

  TensorId inTensorId =
      remoteStoreOp.input->tensor(RemoteStoreOp::getLocalTensorInIndex())->id;

  logging::opx::debug("[RemoteStoreOpx] Growing RemoteStore for tensor {}, "
                      "using RemoteBuffer {}",
                      inTensorId,
                      remoteStoreOp.getRemoteBufferId());

  snap::Tensor inTensor = getInTensor(RemoteStoreOp::getLocalTensorInIndex());

  TensorId offsetId;
  snap::Tensor offset;

  if (remoteStoreOp.input->hasIndex(
          RemoteStoreOp::getRemoteBufferOffsetInIndex())) {
    offsetId = remoteStoreOp.input
                   ->tensor(RemoteStoreOp::getRemoteBufferOffsetInIndex())
                   ->id;
    offset = getInTensor(RemoteStoreOp::getRemoteBufferOffsetInIndex());
  }

  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, remoteStoreOp.getExchangeDescriptor(0));

  descriptorx->setInTensors({{inTensorId, inTensor}, {offsetId, offset}});
  descriptorx->pre(graph(), prog, debugContext());
  descriptorx->exchange(graph(), prog, debugContext());
  descriptorx->post(graph(), prog, debugContext());
}

// RemoteLoadOpx
RemoteLoadOpx::RemoteLoadOpx(Op *op, Devicex *devicex)
    : RemoteBaseOpx(op, devicex) {
  verifyOp<RemoteLoadOp>(op);
}

void RemoteLoadOpx::grow(snap::program::Sequence &prog) const {
  // Obtain the operator
  auto &remoteLoadOp = getOp<RemoteLoadOp>();

  // Obtain the input/output
  TensorId inTensorId =
      remoteLoadOp.input->tensor(RemoteLoadOp::getLocalTensorInIndex())->id;
  snap::Tensor inTensor = getInTensor(RemoteLoadOp::getLocalTensorInIndex());
  TensorId outTensorId =
      remoteLoadOp.output->tensor(RemoteLoadOp::getLocalTensorOutIndex())->id;

  logging::opx::debug("[RemoteLoadOpx] Growing RemoteLoad for tensor {} -> {}, "
                      "using RemoteBuffer {}",
                      inTensorId,
                      outTensorId,
                      remoteLoadOp.getRemoteBufferId());

  // Set the offset tensor (if set)
  TensorId offsetId;
  snap::Tensor offset;
  if (remoteLoadOp.input->hasIndex(
          RemoteLoadOp::getRemoteBufferOffsetInIndex())) {
    offsetId =
        remoteLoadOp.input->tensor(RemoteLoadOp::getRemoteBufferOffsetInIndex())
            ->id;
    offset = getInTensor(RemoteLoadOp::getRemoteBufferOffsetInIndex());
  }

  // Let the exchange descriptor handle the actual loading
  // Whether or not the loading will happen inplace will be handled in the
  // descriptor
  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, remoteLoadOp.getExchangeDescriptor(0));
  descriptorx->setInTensors({{inTensorId, inTensor}, {offsetId, offset}});
  descriptorx->pre(graph(), prog, debugContext());
  descriptorx->exchange(graph(), prog, debugContext());
  descriptorx->post(graph(), prog, debugContext());

  // Set potential view changes
  if (hasInViewChangers(RemoteLoadOp::getLocalTensorInIndex())) {
    setOutViewChangers(
        RemoteLoadOp::getLocalTensorOutIndex(),
        getInViewChangers(RemoteLoadOp::getLocalTensorInIndex()));
  }

  // The output tensor is obtained from the descriptor
  setOutTensor(RemoteLoadOp::getLocalTensorOutIndex(),
               descriptorx->getOutTensors().at(0));
}

InputCreatorType RemoteLoadOpx::getInputCreatorType(InIndex index) const {
  return index == RemoteLoadOp::getLocalTensorInIndex()
             ? InputCreatorType::CanUnwind
             : PopOpx::getInputCreatorType(index);
}

snap::Tensor RemoteLoadOpx::unwindTensorLayout(snap::Tensor tensor,
                                               InIndex in,
                                               OutIndex out) const {
  auto &remoteLoadOp = getOp<RemoteLoadOp>();
  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, remoteLoadOp.getExchangeDescriptor(0));
  return descriptorx->unwind(srcVirtualGraph(in), tensor);
}

view::RegMap RemoteLoadOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

// RemoteLoadInplaceOpx
RemoteLoadInplaceOpx::RemoteLoadInplaceOpx(Op *op, Devicex *devicex)
    : RemoteLoadOpx(op, devicex) {
  verifyOp<RemoteLoadInplaceOp>(op, Onnx::CustomOperators::RemoteLoadInplace);
}

namespace {
OpxCreator<RemoteStoreOpx>
    remoteStoreOpxCreator(Onnx::CustomOperators::RemoteStore);
OpxCreator<RemoteLoadOpx>
    remoteLoadOpxCreator(Onnx::CustomOperators::RemoteLoad);
OpxCreator<RemoteLoadInplaceOpx>
    remoteLoadInplaceOpxCreator(Onnx::CustomOperators::RemoteLoadInplace);
} // namespace
} // namespace popx
} // namespace popart
