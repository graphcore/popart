// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/hostcopy.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/hostcopyx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

HostBaseOpx::HostBaseOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

poplar::Tensor HostBaseOpx::load(poplar::program::Sequence &prog,
                                 const TensorId &outTensorId,
                                 const TensorId &streamTensorId) const {
  auto streams = dv_p->lowering().getFromHostStreams();

  auto it = streams.find(streamTensorId);

  if (it != streams.end()) {
    logging::opx::debug("Found host stream in getFromHostStreams {}",
                        streamTensorId);
    auto stream      = streams.at(streamTensorId);
    poplar::Tensor t = get(outTensorId).getPoplarTensor();

    if (!t.isParallelWriteable()) {
      logging::opx::debug("Tensor {} is not a writable host load tensor "
                          " target, cloning. "
                          "The aliasing properties have changed implicitly.",
                          outTensorId);
      poplar::Tensor tw = graph().getPoplarGraph().clone(
          t,
          debugContext(outTensorId + "_Writable"),
          poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
      poplar::program::Copy copy_prog(
          stream, tw, false, {outTensorId + "_copy_clone"});
      prog.add(copy_prog);

    } else {
      poplar::program::Copy copy_prog(
          stream, t, false, {outTensorId + "_copy"});
      prog.add(copy_prog);
    }
    return t;

  } else {
    throw error("Stream for from host op {}, tensor {} not found",
                op_p->debugName(),
                streamTensorId);
  }
  return poplar::Tensor();
}

void HostBaseOpx::store(poplar::program::Sequence &prog,
                        const TensorId &inTensorId,
                        const TensorId &streamTensorId) const {
  poplar::Tensor t;
  ToHostStreamType stype;
  auto &ir = op_p->getIr();

  switch (ir.getDataFlow().art(streamTensorId).id()) {
  // Copy program runs after every batch
  case (AnchorReturnTypeId::All): {
    stype = ToHostStreamType::NonSumAnchor;
    break;
  }
  // Copy program runs at the end of every N batches
  case (AnchorReturnTypeId::EveryN): {
    if (ir.getSessionOptions().enablePipelining) {
      throw error("AnchorReturnType::EVERYN is not valid for pipelined models");
    } else {
      throw error("AnchorReturnType::EVERYN not supported yet");
    }
    break;
  }
  // Copy program runs at the end of the step
  case (AnchorReturnTypeId::Final): {
    stype = ToHostStreamType::NonSumAnchor;
    break;
  }
  case (AnchorReturnTypeId::Sum): {
    stype = ToHostStreamType::SumAnchor;
    break;
  }
  }

  auto streams = dv_p->lowering().getToHostAnchorStreams();

  if (stype == ToHostStreamType::NonAnchor) {
    streams = dv_p->lowering().getToHostWeightStreams();
  }

  auto it = streams.find(streamTensorId);

  if (it != streams.end()) {
    logging::opx::debug("Found host stream in getFromHostStreams {}",
                        streamTensorId);
    auto stream = streams.at(streamTensorId);

    auto tensor = op_p->input->tensor(HostStoreOp::getLocalTensorInIndex());
    const auto &anchorTensor =
        stype == ToHostStreamType::SumAnchor
            ? dv_p->lowering().tensors().getView(anchorSumPrefix() + tensor->id)
            : dv_p->lowering().tensors().getView(tensor->id);

    auto nElmsStream = stream.numElements();
    auto nElmsTensor = anchorTensor.getPoplarTensor().numElements();
    if (nElmsStream != nElmsTensor) {
      throw internal_error("[Devicex::toHostTask] "
                           "The poplar::Tensor {} has {}, whereas the "
                           "poplar::Stream has {}. These should be the same.",
                           tensor->id,
                           nElmsTensor,
                           nElmsStream);
    }

    poplar::program::Copy copy_prog(
        anchorTensor.getPoplarTensor(), stream, false, {inTensorId + "_copy"});
    prog.add(copy_prog);

  } else {
    throw error("Stream for to host op {}, tensor {} not found",
                op_p->debugName(),
                streamTensorId);
  }
}

HostLoadOpx::HostLoadOpx(Op *op, Devicex *devicex) : HostBaseOpx(op, devicex) {
  verifyOp<HostLoadOp>(op, Onnx::CustomOperators::HostLoad);
}

void HostLoadOpx::grow(poplar::program::Sequence &prog) const {
  auto &hostLoadOp = getOp<HostLoadOp>();

  TensorId inTensorId =
      hostLoadOp.input->tensor(HostLoadOp::getLocalTensorInIndex())->id;

  logging::opx::debug(
      "[HostLoadOpx] Growing HostLoad for tensor {} -> {}, "
      "using Stream {}",
      hostLoadOp.input->tensor(HostLoadOp::getLocalTensorInIndex())->id,
      hostLoadOp.output->tensor(HostLoadOp::getLocalTensorOutIndex())->id,
      hostLoadOp.getHostStreamTensorId());

  auto t = load(prog, inTensorId, hostLoadOp.getHostStreamTensorId());

  setOutTensor(HostLoadOp::getLocalTensorOutIndex(), snap::Tensor{t, graph()});

  if (hasInViewChangers(HostLoadOp::getLocalTensorInIndex())) {
    setOutViewChangers(HostLoadOp::getLocalTensorOutIndex(),
                       getInViewChangers(HostLoadOp::getLocalTensorInIndex()));
  }
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

  logging::opx::debug(
      "[HostStoreOpx] Growing HostStore for tensor {} using Stream {}",
      hostStoreOp.input->tensor(HostStoreOp::getLocalTensorInIndex())->id,
      hostStoreOp.getHostStreamTensorId());

  store(prog, inTensorId, hostStoreOp.getHostStreamTensorId());
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
