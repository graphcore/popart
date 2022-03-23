// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/exchange/hostcopyx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

HostBaseOpx::HostBaseOpx(Op *op, Devicex *devicex)
    : ExchangeBaseOpx(op, devicex) {}

HostLoadOpx::HostLoadOpx(Op *op, Devicex *devicex) : HostBaseOpx(op, devicex) {
  verifyOp<HostLoadOp>(op);
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void HostLoadOpx::grow(snap::program::Sequence &prog) const {
  auto &hostLoadOp = getOp<HostLoadOp>();

  TensorId inTensorId =
      hostLoadOp.input->tensor(HostLoadOp::getLocalTensorInIndex())->id;
  snap::Tensor inTensor = getInTensor(HostLoadOp::getLocalTensorInIndex());

  logging::opx::debug(
      "[HostLoadOpx] Growing HostLoad for tensor {} -> {}, "
      "using Stream {}",
      hostLoadOp.input->tensor(HostLoadOp::getLocalTensorInIndex())->id,
      hostLoadOp.output->tensor(HostLoadOp::getLocalTensorOutIndex())->id,
      hostLoadOp.getHostStreamTensorId());

  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, hostLoadOp.getExchangeDescriptor(0));

  descriptorx->setInTensors({{inTensorId, inTensor}});
  descriptorx->pre(graph(), prog, debugContext());
  descriptorx->exchange(graph(), prog, debugContext());
  descriptorx->post(graph(), prog, debugContext());

  if (hasInViewChangers(HostLoadOp::getLocalTensorInIndex())) {
    setOutViewChangers(HostLoadOp::getLocalTensorOutIndex(),
                       getInViewChangers(HostLoadOp::getLocalTensorInIndex()));
  }

  setOutTensor(HostLoadOp::getLocalTensorOutIndex(),
               descriptorx->getOutTensors().at(0));
}

InputCreatorType HostLoadOpx::getInputCreatorType(InIndex index) const {
  auto &hostLoadOp = getOp<HostLoadOp>();
  auto descriptor  = hostLoadOp.getExchangeDescriptor(0);
  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, hostLoadOp.getExchangeDescriptor(0));
  if (descriptorx->rearrangeOnHost() ||
      descriptor.getTileSet() == TileSet::Compute) {
    // If rearranging on host or not using IO tiles, then use unwinding to
    // minimize rearrangements

    // `Unwind`: In most cases, the input tensor layout can be unwound from the
    // output, which will cause fewer on-device rearrangements
    return InputCreatorType::CanUnwind;
  } else {
    // If rearranging on device and using IO tiles, use host transferrable
    // tensors to facilitate overlapped IO/compute

    // `CanCreate`: Create the tensor with createHostTransferrableTensor to
    // avoid blocking overlapped IO with misplaced inter-tile exchanges on
    // IO tiles
    // `Unwind`: Fallback if creating the tensor is not possible
    return InputCreatorType::CanCreateOrUnwind;
  }
  return PopOpx::getInputCreatorType(index);
}

snap::Tensor
HostLoadOpx::createInputTensor(InIndex index,
                               const poplar::DebugNameAndId &dnai) const {
  auto &hostLoadOp = getOp<HostLoadOp>();
  auto descriptor  = hostLoadOp.getExchangeDescriptor(0);
  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, descriptor);
  return descriptorx->create(inGraph(index), inInfo(index));
}

snap::Tensor HostLoadOpx::unwindTensorLayout(snap::Tensor tensor,
                                             InIndex in,
                                             OutIndex out) const {
  auto &hostLoadOp = getOp<HostLoadOp>();
  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, hostLoadOp.getExchangeDescriptor(0));
  return descriptorx->unwind(srcVirtualGraph(in), tensor);
}

view::RegMap HostLoadOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

// RemoteLoadInplaceOpx
HostLoadInplaceOpx::HostLoadInplaceOpx(Op *op, Devicex *devicex)
    : HostLoadOpx(op, devicex) {
  verifyOp<HostLoadInplaceOp>(op, Onnx::CustomOperators::HostLoadInplace);
}

HostStoreOpx::HostStoreOpx(Op *op, Devicex *devicex)
    : HostBaseOpx(op, devicex) {
  verifyOp<HostStoreOp>(op, Onnx::CustomOperators::HostStore);
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void HostStoreOpx::grow(snap::program::Sequence &prog) const {
  auto &hostStoreOp = getOp<HostStoreOp>();

  TensorId inTensorId =
      hostStoreOp.input->tensor(HostStoreOp::getLocalTensorInIndex())->id;
  snap::Tensor inTensor = getInTensor(HostStoreOp::getLocalTensorInIndex());

  logging::opx::debug(
      "[HostStoreOpx] Growing HostStore for tensor {} using Stream {}",
      hostStoreOp.input->tensor(HostStoreOp::getLocalTensorInIndex())->id,
      hostStoreOp.getHostStreamTensorId());

  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, hostStoreOp.getExchangeDescriptor(0));

  descriptorx->setInTensors({{inTensorId, inTensor}});
  descriptorx->pre(graph(), prog, debugContext());
  descriptorx->exchange(graph(), prog, debugContext());
  descriptorx->post(graph(), prog, debugContext());
}

InputCreatorType HostStoreOpx::getInputCreatorType(InIndex index) const {
  auto &hostStoreOp = getOp<HostStoreOp>();
  auto descriptor   = hostStoreOp.getExchangeDescriptor(0);
  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, descriptor);
  if (descriptorx->rearrangeOnHost() ||
      descriptor.getTileSet() == TileSet::Compute) {
    // `Deadend`: HostStore has no output tensor, therefore unwinding is not
    // available
    return InputCreatorType::Deadend;
  } else {
    // If rearranging on device and using IO tiles, use host transferrable
    // tensors to facilitate overlapped IO/compute

    // `CanCreate`: Create the tensor with createHostTransferrableTensor to
    // avoid blocking overlapped IO with misplaced inter-tile exchanges on
    // IO tiles
    return InputCreatorType::CanCreate;
  }
  return PopOpx::getInputCreatorType(index);
}

snap::Tensor
HostStoreOpx::createInputTensor(InIndex index,
                                const poplar::DebugNameAndId &dnai) const {
  auto &hostStoreOp = getOp<HostStoreOp>();
  auto descriptor   = hostStoreOp.getExchangeDescriptor(0);
  std::shared_ptr<ExchangeDescriptorx> descriptorx =
      getExchangeDescriptorx(dv_p, descriptor);
  return descriptorx->create(inGraph(index), inInfo(index));
}

namespace {
OpxCreator<HostLoadOpx> hostLoadOpxCreator(Onnx::CustomOperators::HostLoad);
OpxCreator<HostLoadInplaceOpx>
    hostLoadInplaceOpxCreator(Onnx::CustomOperators::HostLoadInplace);
OpxCreator<HostStoreOpx> hostStoreOpxCreator(Onnx::CustomOperators::HostStore);
} // namespace
} // namespace popx
} // namespace popart
