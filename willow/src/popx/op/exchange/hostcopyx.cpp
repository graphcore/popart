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
  verifyOp<HostLoadOp>(op, Onnx::CustomOperators::HostLoad);
}

void HostLoadOpx::grow(poplar::program::Sequence &prog) const {
  auto &hostLoadOp = getOp<HostLoadOp>();

  TensorId inTensorId =
      hostLoadOp.input->tensor(HostLoadOp::getLocalTensorInIndex())->id;
  poplar::Tensor inTensor =
      getInTensor(HostLoadOp::getLocalTensorInIndex()).getPoplarTensor();

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
               snap::Tensor{descriptorx->getOutTensors().at(0), graph()});
}

InputCreatorType HostLoadOpx::getInputCreatorType(InIndex index) const {
  return index == HostLoadOp::getLocalTensorInIndex()
             ? InputCreatorType::CanUnwind
             : PopOpx::getInputCreatorType(index);
}

snap::Tensor
HostLoadOpx::unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

view::RegMap HostLoadOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

HostStoreOpx::HostStoreOpx(Op *op, Devicex *devicex)
    : HostBaseOpx(op, devicex) {
  verifyOp<HostStoreOp>(op, Onnx::CustomOperators::HostStore);
}

void HostStoreOpx::grow(poplar::program::Sequence &prog) const {
  auto &hostStoreOp = getOp<HostStoreOp>();

  TensorId inTensorId =
      hostStoreOp.input->tensor(HostStoreOp::getLocalTensorInIndex())->id;
  poplar::Tensor inTensor =
      getInTensor(HostStoreOp::getLocalTensorInIndex()).getPoplarTensor();

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
  return index == HostStoreOp::getLocalTensorInIndex()
             ? InputCreatorType::CanUnwind
             : PopOpx::getInputCreatorType(index);
}

snap::Tensor
HostStoreOpx::unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

view::RegMap HostStoreOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

namespace {
OpxCreator<HostLoadOpx> hostLoadOpxCreator(Onnx::CustomOperators::HostLoad);
OpxCreator<HostStoreOpx> hostStoreOpxCreator(Onnx::CustomOperators::HostStore);
} // namespace
} // namespace popx
} // namespace popart
