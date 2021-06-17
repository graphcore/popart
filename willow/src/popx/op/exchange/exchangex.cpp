// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/op/exchange/multiexchangex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

namespace {
poplar::Tensor makeWritableRemoteExchangeTensor(Devicex *dv_p,
                                                TensorId id,
                                                RemoteBufferId rbid,
                                                snap::Graph &graph,
                                                poplar::program::Sequence &prog,
                                                poplar::Tensor t) {
  poplar::Tensor rbTensor;
  if (!dv_p->lowering().hasRemoteBuffer(rbid)) {
    rbTensor = graph.getPoplarGraph().clone(
        t,
        {"RB_" + std::to_string(rbid)},
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().createRemoteBuffer(rbid, rbTensor);
  }
  auto buffer = dv_p->lowering().getRemoteBuffer(rbid);
  rbTensor    = buffer.second.value();
  if (!t.isParallelWriteable() || t.containsConstant()) {
    logging::opx::warn("Tensor {} is not a writable remote buffer "
                       "copy target, cloning. "
                       "The aliasing properties have changed implicitly.",
                       id);
    poplar::Tensor tw = graph.getPoplarGraph().clone(
        rbTensor,
        {id + "_writable"},
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    return tw;
  }
  return t;
}

poplar::Tensor makeWritableHostExchangeTensor(Devicex *dv_p,
                                              TensorId id,
                                              TensorId streamTensorId,
                                              snap::Graph &graph,
                                              poplar::program::Sequence &prog,
                                              poplar::Tensor t) {
  poplar::Tensor streamTensor;
  if (dv_p->lowering().tensors().contains(streamTensorId)) {
    streamTensor = dv_p->lowering().tensors().get(streamTensorId);
  } else {
    streamTensor = graph.getPoplarGraph().clone(
        t,
        {streamTensorId},
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().tensors().insert(streamTensorId, streamTensor);
  }
  if (!t.isParallelWriteable()) {
    logging::opx::debug("Tensor {} is not a writable host load tensor "
                        " target, cloning. "
                        "The aliasing properties have changed implicitly.",
                        id);
    poplar::Tensor tw = graph.getPoplarGraph().clone(
        streamTensor,
        {id + "_writable"},
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    return tw;
  }
  return t;
}

poplar::Tensor getOrCreateStreamTensor(Devicex *dv_p,
                                       const ExchangeDescriptor &descriptor,
                                       snap::Graph &graph,
                                       poplar::Tensor t,
                                       const poplar::DebugContext &context) {
  poplar::Tensor streamTensor;

  if (dv_p->lowering().hasStreamTensor("ST_" +
                                       descriptor.getHostStreamTensorId())) {
    streamTensor = dv_p->lowering().getStreamTensor(
        "ST_" + descriptor.getHostStreamTensorId());
  } else {
    streamTensor = graph.getPoplarGraph().clone(
        t,
        poplar::DebugContext(context, descriptor.getHostStreamTensorId()),
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().setStreamTensor("ST_" + descriptor.getHostStreamTensorId(),
                                     streamTensor);
  }
  return streamTensor;
}
} // namespace

ExchangeBaseOpx::ExchangeBaseOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {}

ExchangeDescriptorx::ExchangeDescriptorx(Devicex *dv_p_,
                                         ExchangeDescriptor descriptor_)
    : dv_p(dv_p_), descriptor(descriptor_) {}

std::shared_ptr<ExchangeDescriptorx>
getExchangeDescriptorx(Devicex *dv_p, ExchangeDescriptor descriptor) {
  if (descriptor.isHostExchange()) {
    switch (descriptor.getDirection()) {
    case ExchangeDirection::Load: {
      return std::make_unique<HostLoadDescriptorx>(dv_p, descriptor);
      break;
    }
    case ExchangeDirection::Store: {
      return std::make_unique<HostStoreDescriptorx>(dv_p, descriptor);
      break;
    }
    default:
      throw internal_error("Unexpected exchange direction {}",
                           descriptor.getDirection());
    }
  } else if (descriptor.isRemoteExchange()) {
    switch (descriptor.getDirection()) {
    case ExchangeDirection::Load: {
      return std::make_unique<RemoteLoadDescriptorx>(dv_p, descriptor);
      break;
    }
    case ExchangeDirection::Store: {
      return std::make_unique<RemoteStoreDescriptorx>(dv_p, descriptor);
      break;
    }
    default:
      throw internal_error("Unexpected exchange direction {}",
                           descriptor.getDirection());
    }
  } else {
    throw internal_error("Unexpected exchange descriptor.");
  }
}

HostLoadDescriptorx::HostLoadDescriptorx(Devicex *dv_p_,
                                         ExchangeDescriptor descriptor_)
    : ExchangeDescriptorx(dv_p_, descriptor_) {}

HostStoreDescriptorx::HostStoreDescriptorx(Devicex *dv_p_,
                                           ExchangeDescriptor descriptor_)
    : ExchangeDescriptorx(dv_p_, descriptor_) {}

RemoteLoadDescriptorx::RemoteLoadDescriptorx(Devicex *dv_p_,
                                             ExchangeDescriptor descriptor_)
    : ExchangeDescriptorx(dv_p_, descriptor_) {}

RemoteStoreDescriptorx::RemoteStoreDescriptorx(Devicex *dv_p_,
                                               ExchangeDescriptor descriptor_)
    : ExchangeDescriptorx(dv_p_, descriptor_) {}

void HostLoadDescriptorx::pre(snap::Graph &graph,
                              poplar::program::Sequence &prog,
                              poplar::DebugContext context) {}

void HostLoadDescriptorx::exchange(snap::Graph &graph,
                                   poplar::program::Sequence &prog,
                                   poplar::DebugContext context) {
  auto streamTensor = getOrCreateStreamTensor(
      dv_p, descriptor, graph, inTensors.at(0).second, context);

  auto streams = dv_p->lowering().getFromHostStreams();

  auto it = streams.find(descriptor.getHostStreamTensorId());

  if (it != streams.end()) {
    logging::opx::debug("Found host stream in getFromHostStreams {}",
                        descriptor.getHostStreamTensorId());
    auto stream = streams.at(descriptor.getHostStreamTensorId());

    poplar::program::Copy copy_prog(stream, streamTensor, false, context);
    prog.add(copy_prog);

  } else {
    throw error("Stream for tensor {} not found",
                descriptor.getHostStreamTensorId());
  }
}

void HostLoadDescriptorx::post(snap::Graph &graph,
                               poplar::program::Sequence &prog,
                               poplar::DebugContext context) {
  outTensors.push_back(
      makeWritableHostExchangeTensor(dv_p,
                                     inTensors.at(0).first,
                                     descriptor.getHostStreamTensorId(),
                                     graph,
                                     prog,
                                     inTensors.at(0).second));

  poplar::Tensor streamTensor = getOrCreateStreamTensor(
      dv_p, descriptor, graph, outTensors.at(0), context);
  poplar::program::Copy tmp_copy_prog(
      streamTensor, outTensors.at(0), false, context);
  prog.add(tmp_copy_prog);
}

void HostStoreDescriptorx::pre(snap::Graph &graph,
                               poplar::program::Sequence &prog,
                               poplar::DebugContext context) {
  auto streamTensor = getOrCreateStreamTensor(
      dv_p, descriptor, graph, inTensors.at(0).second, context);

  poplar::program::Copy tmp_copy_prog(
      inTensors.at(0).second, streamTensor, false, context);
  prog.add(tmp_copy_prog);
}

void HostStoreDescriptorx::exchange(snap::Graph &graph,
                                    poplar::program::Sequence &prog,
                                    poplar::DebugContext context) {
  auto streams = dv_p->lowering().getToHostAnchorStreams();
  auto it      = streams.find(descriptor.getHostStreamTensorId());

  poplar::Tensor streamTensor = getOrCreateStreamTensor(
      dv_p, descriptor, graph, inTensors.at(0).second, context);

  if (it != streams.end()) {
    logging::opx::debug("Found host stream in getFromHostStreams {}",
                        descriptor.getHostStreamTensorId());
    auto stream      = streams.at(descriptor.getHostStreamTensorId());
    auto nElmsStream = stream.numElements();
    auto nElmsTensor = streamTensor.numElements();
    if (nElmsStream != nElmsTensor) {
      throw internal_error("[Devicex::toHostTask] "
                           "The poplar::Tensor {} has {}, whereas the "
                           "poplar::Stream has {}. These should be the same.",
                           inTensors.at(0).first,
                           nElmsTensor,
                           nElmsStream);
    }

    poplar::program::Copy copy_prog(streamTensor, stream, false, context);
    prog.add(copy_prog);

  } else {
    throw error("Stream for tensor {} not found",
                descriptor.getHostStreamTensorId());
  }
}

void HostStoreDescriptorx::post(snap::Graph &graph,
                                poplar::program::Sequence &prog,
                                poplar::DebugContext context) {}

void RemoteLoadDescriptorx::pre(snap::Graph &graph,
                                poplar::program::Sequence &prog,
                                poplar::DebugContext context) {}

void RemoteLoadDescriptorx::exchange(snap::Graph &graph,
                                     poplar::program::Sequence &prog,
                                     poplar::DebugContext context) {
  poplar::Tensor rbTensor;

  poplar::Tensor offset;
  if (inTensors.size() > 1) {
    offset = inTensors.at(1).second;
  }

  if (!dv_p->lowering().hasRemoteBuffer(descriptor.getRemoteBufferId())) {
    rbTensor = graph.getPoplarGraph().clone(
        inTensors.at(0).second,
        poplar::DebugContext(context,
                             dv_p->lowering().getRemoteBufferName(
                                 descriptor.getRemoteBufferId())),
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().createRemoteBuffer(descriptor.getRemoteBufferId(),
                                        rbTensor);
  }

  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  rbTensor = buffer.second.value();

  if (offset.valid() && offset.numElements() > 0) {
    poplar::program::Copy copy_prog(buffer.first, rbTensor, offset, context);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(buffer.first, rbTensor, context);
    prog.add(copy_prog);
  }
}

void RemoteLoadDescriptorx::post(snap::Graph &graph,
                                 poplar::program::Sequence &prog,
                                 poplar::DebugContext context) {
  outTensors.push_back(
      makeWritableRemoteExchangeTensor(dv_p,
                                       inTensors.at(0).first,
                                       descriptor.getRemoteBufferId(),
                                       graph,
                                       prog,
                                       inTensors.at(0).second));

  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  poplar::Tensor rbTensor = buffer.second.value();
  poplar::program::Copy tmp_copy_prog(
      rbTensor, outTensors.at(0), false, context);
  prog.add(tmp_copy_prog);
}

void RemoteStoreDescriptorx::pre(snap::Graph &graph,
                                 poplar::program::Sequence &prog,
                                 poplar::DebugContext context) {
  poplar::Tensor rbTensor;
  if (!dv_p->lowering().hasRemoteBuffer(descriptor.getRemoteBufferId())) {
    rbTensor = graph.getPoplarGraph().clone(
        inTensors.at(0).second,
        poplar::DebugContext(context,
                             dv_p->lowering().getRemoteBufferName(
                                 descriptor.getRemoteBufferId())),
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().createRemoteBuffer(descriptor.getRemoteBufferId(),
                                        rbTensor);
  }
  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  rbTensor = buffer.second.value();
  poplar::program::Copy tmp_copy_prog(
      inTensors.at(0).second, rbTensor, false, context);
  prog.add(tmp_copy_prog);
}

void RemoteStoreDescriptorx::exchange(snap::Graph &graph,
                                      poplar::program::Sequence &prog,
                                      poplar::DebugContext context) {
  poplar::Tensor offset;
  if (inTensors.size() > 1) {
    offset = inTensors.at(1).second;
  }

  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  poplar::Tensor rbTensor = buffer.second.value();
  if (offset.valid() && offset.numElements() > 0) {
    poplar::program::Copy copy_prog(rbTensor, buffer.first, offset, context);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(rbTensor, buffer.first, context);
    prog.add(copy_prog);
  }
}

void RemoteStoreDescriptorx::post(snap::Graph &graph,
                                  poplar::program::Sequence &prog,
                                  poplar::DebugContext context) {}

} // namespace popx
} // namespace popart
