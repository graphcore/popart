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
snap::Tensor makeWritableRemoteExchangeTensor(Devicex *dv_p,
                                              TensorId id,
                                              RemoteBufferId rbid,
                                              snap::Graph &graph,
                                              poplar::program::Sequence &prog,
                                              snap::Tensor t) {
  snap::Tensor rbTensor;
  if (!dv_p->lowering().hasRemoteBuffer(rbid)) {
    rbTensor = snap::Tensor{
        graph.getPoplarGraph().clone(
            t.getPoplarTensor(),
            {"RB_" + std::to_string(rbid)},
            poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES),
        graph};
    dv_p->lowering().createRemoteBuffer(rbid, rbTensor);
  }
  auto buffer = dv_p->lowering().getRemoteBuffer(rbid);
  rbTensor    = buffer.second.value();
  if (!t.getPoplarTensor().isParallelWriteable() ||
      t.getPoplarTensor().containsConstant()) {
    logging::opx::warn("Tensor {} is not a writable remote buffer "
                       "copy target, cloning. "
                       "The aliasing properties have changed implicitly.",
                       id);
    snap::Tensor tw =
        snap::Tensor{graph.getPoplarGraph().clone(
                         rbTensor.getPoplarTensor(),
                         {id + "_writable"},
                         poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES),
                     graph};
    return tw;
  }
  return t;
}

snap::Tensor getOrCreateStreamTensor(Devicex *dv_p,
                                     TensorId streamTensorId,
                                     snap::Graph &graph,
                                     snap::Tensor t,
                                     const poplar::DebugContext &context) {
  snap::Tensor streamTensor;

  if (dv_p->lowering().hasStreamTensor("ST_" + streamTensorId)) {
    streamTensor = dv_p->lowering().getStreamTensor("ST_" + streamTensorId);
  } else {
    streamTensor = snap::Tensor{
        graph.getPoplarGraph().clone(
            t.getPoplarTensor(),
            poplar::DebugContext(context, streamTensorId),
            poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES),
        graph};
    dv_p->lowering().setStreamTensor("ST_" + streamTensorId, streamTensor);
  }
  return streamTensor;
}

snap::Tensor
makeWritableHostExchangeTensor(Devicex *dv_p,
                               TensorId id,
                               TensorId streamTensorId,
                               snap::Graph &graph,
                               poplar::program::Sequence &prog,
                               snap::Tensor t,
                               const poplar::DebugContext &context) {
  snap::Tensor streamTensor =
      getOrCreateStreamTensor(dv_p, id, graph, t, context);
  if (dv_p->lowering().tensors().contains(streamTensorId)) {
    streamTensor = dv_p->lowering().tensors().get(streamTensorId);
  } else {
    streamTensor = snap::Tensor{
        graph.getPoplarGraph().clone(
            t.getPoplarTensor(),
            {streamTensorId},
            poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES),
        graph};
    dv_p->lowering().setStreamTensor(id, streamTensor);
  }
  if (!t.getPoplarTensor().isParallelWriteable()) {
    logging::opx::debug("Tensor {} is not a writable host load tensor "
                        " target, cloning. "
                        "The aliasing properties have changed implicitly.",
                        id);
    snap::Tensor tw = snap::Tensor{
        graph.getPoplarGraph().clone(
            streamTensor.getPoplarTensor(),
            {id + "_writable"},
            poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES),
        graph};
    return tw;
  }
  return t;
}

} // namespace

ExchangeBaseOpx::ExchangeBaseOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {}

ExchangeDescriptorx::ExchangeDescriptorx(Devicex *dv_p_,
                                         ExchangeDescriptor descriptor_)
    : dv_p(dv_p_), descriptor(descriptor_) {}

std::unique_ptr<ExchangeDescriptorx>
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
  auto streamTensor =
      getOrCreateStreamTensor(dv_p,
                              descriptor.getHostStreamTensorId(),
                              graph,
                              inTensors.at(0).second,
                              context);

  auto streams = dv_p->lowering().getFromHostStreams();

  auto it = streams.find(descriptor.getHostStreamTensorId());

  if (it != streams.end()) {
    logging::opx::debug("Found host stream in getFromHostStreams {}",
                        descriptor.getHostStreamTensorId());
    auto stream = streams.at(descriptor.getHostStreamTensorId());

    poplar::program::Copy copy_prog(
        stream, streamTensor.getPoplarTensor(), false, context);
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
                                     inTensors.at(0).second,
                                     context));

  snap::Tensor streamTensor =
      getOrCreateStreamTensor(dv_p,
                              descriptor.getHostStreamTensorId(),
                              graph,
                              outTensors.at(0),
                              context);
  poplar::program::Copy tmp_copy_prog(streamTensor.getPoplarTensor(),
                                      outTensors.at(0).getPoplarTensor(),
                                      false,
                                      context);
  prog.add(tmp_copy_prog);
}

void HostStoreDescriptorx::pre(snap::Graph &graph,
                               poplar::program::Sequence &prog,
                               poplar::DebugContext context) {
  auto streamTensor =
      getOrCreateStreamTensor(dv_p,
                              descriptor.getHostStreamTensorId(),
                              graph,
                              inTensors.at(0).second,
                              context);

  poplar::program::Copy tmp_copy_prog(inTensors.at(0).second.getPoplarTensor(),
                                      streamTensor.getPoplarTensor(),
                                      false,
                                      context);
  prog.add(tmp_copy_prog);
}

void HostStoreDescriptorx::exchange(snap::Graph &graph,
                                    poplar::program::Sequence &prog,
                                    poplar::DebugContext context) {
  auto streams = dv_p->lowering().getToHostAnchorStreams();
  auto it      = streams.find(descriptor.getHostStreamTensorId());

  snap::Tensor streamTensor =
      getOrCreateStreamTensor(dv_p,
                              descriptor.getHostStreamTensorId(),
                              graph,
                              inTensors.at(0).second,
                              context);

  if (it != streams.end()) {
    logging::opx::debug("Found host stream in getFromHostStreams {}",
                        descriptor.getHostStreamTensorId());
    auto stream      = streams.at(descriptor.getHostStreamTensorId());
    auto nElmsStream = stream.numElements();
    auto nElmsTensor = streamTensor.getPoplarTensor().numElements();
    if (nElmsStream != nElmsTensor) {
      throw internal_error("[Devicex::toHostTask] "
                           "The poplar::Tensor {} has {}, whereas the "
                           "poplar::Stream has {}. These should be the same.",
                           inTensors.at(0).first,
                           nElmsTensor,
                           nElmsStream);
    }

    poplar::program::Copy copy_prog(
        streamTensor.getPoplarTensor(), stream, false, context);
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
  snap::Tensor rbTensor;

  snap::Tensor offset;
  if (inTensors.size() > 1) {
    offset = inTensors.at(1).second;
  }

  if (!dv_p->lowering().hasRemoteBuffer(descriptor.getRemoteBufferId())) {
    rbTensor = snap::Tensor{
        graph.getPoplarGraph().clone(
            inTensors.at(0).second.getPoplarTensor(),
            poplar::DebugContext(context,
                                 dv_p->lowering().getRemoteBufferName(
                                     descriptor.getRemoteBufferId())),
            poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES),
        graph};
    dv_p->lowering().createRemoteBuffer(descriptor.getRemoteBufferId(),
                                        rbTensor);
  }

  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  rbTensor = buffer.second.value();

  if (offset.valid() && offset.getPoplarTensor().numElements() > 0) {
    poplar::program::Copy copy_prog(buffer.first,
                                    rbTensor.getPoplarTensor(),
                                    offset.getPoplarTensor(),
                                    context);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(
        buffer.first, rbTensor.getPoplarTensor(), context);
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
  snap::Tensor rbTensor = buffer.second.value();
  poplar::program::Copy tmp_copy_prog(rbTensor.getPoplarTensor(),
                                      outTensors.at(0).getPoplarTensor(),
                                      false,
                                      context);
  prog.add(tmp_copy_prog);
}

void RemoteStoreDescriptorx::pre(snap::Graph &graph,
                                 poplar::program::Sequence &prog,
                                 poplar::DebugContext context) {
  snap::Tensor rbTensor;
  if (!dv_p->lowering().hasRemoteBuffer(descriptor.getRemoteBufferId())) {
    rbTensor = snap::Tensor{
        graph.getPoplarGraph().clone(
            inTensors.at(0).second.getPoplarTensor(),
            poplar::DebugContext(context,
                                 dv_p->lowering().getRemoteBufferName(
                                     descriptor.getRemoteBufferId())),
            poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES),
        graph};
    dv_p->lowering().createRemoteBuffer(descriptor.getRemoteBufferId(),
                                        rbTensor);
  }
  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  rbTensor = buffer.second.value();
  poplar::program::Copy tmp_copy_prog(inTensors.at(0).second.getPoplarTensor(),
                                      rbTensor.getPoplarTensor(),
                                      false,
                                      context);
  prog.add(tmp_copy_prog);
}

void RemoteStoreDescriptorx::exchange(snap::Graph &graph,
                                      poplar::program::Sequence &prog,
                                      poplar::DebugContext context) {
  snap::Tensor offset;
  if (inTensors.size() > 1) {
    offset = inTensors.at(1).second;
  }

  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  snap::Tensor rbTensor = buffer.second.value();
  if (offset.valid() && offset.getPoplarTensor().numElements() > 0) {
    poplar::program::Copy copy_prog(rbTensor.getPoplarTensor(),
                                    buffer.first,
                                    offset.getPoplarTensor(),
                                    context);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(
        rbTensor.getPoplarTensor(), buffer.first, context);
    prog.add(copy_prog);
  }
}

void RemoteStoreDescriptorx::post(snap::Graph &graph,
                                  poplar::program::Sequence &prog,
                                  poplar::DebugContext context) {}

} // namespace popx
} // namespace popart
