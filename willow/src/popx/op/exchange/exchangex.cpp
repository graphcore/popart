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
                                              snap::Tensor t,
                                              bool inplace) {
  snap::Tensor rbTensor;
  // Clone the input tensor if the remote buffer tensor does not exist
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
  // The outplacing of the tensor will be done explicitly
  if (!inplace) {
    snap::Tensor tw =
        snap::Tensor{graph.getPoplarGraph().clone(
                         rbTensor.getPoplarTensor(),
                         {id + "_writable"},
                         poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES),
                     graph};
    return tw;
  }
  // Outplace fallback
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

snap::Tensor ExchangeDescriptorx::unwind(snap::Graph &,
                                         snap::Tensor tensor) const {
  return tensor;
}

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
                              snap::program::Sequence &prog,
                              poplar::DebugContext context) {}

void HostLoadDescriptorx::exchange(snap::Graph &graph,
                                   snap::program::Sequence &prog,
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
    bool rearrangeOnHost = dv_p->lowering().doRearrangeOnHost(
        dv_p->ir().getTensor(descriptor.getHostStreamTensorId()));
    logging::opx::debug(
        "Found host stream in getFromHostStreams {} doRearrangeOnHost: {}",
        descriptor.getHostStreamTensorId(),
        rearrangeOnHost);
    auto stream = streams.at(descriptor.getHostStreamTensorId());

    snap::program::Copy copy_prog(
        stream, streamTensor, rearrangeOnHost, context);
    prog.add(copy_prog);

  } else {
    throw error("Stream for tensor {} not found",
                descriptor.getHostStreamTensorId());
  }
}

void HostLoadDescriptorx::post(snap::Graph &graph,
                               snap::program::Sequence &prog,
                               poplar::DebugContext context) {
  outTensors.push_back(
      makeWritableHostExchangeTensor(dv_p,
                                     inTensors.at(0).first,
                                     descriptor.getHostStreamTensorId(),
                                     graph,
                                     inTensors.at(0).second,
                                     context));

  snap::Tensor streamTensor =
      getOrCreateStreamTensor(dv_p,
                              descriptor.getHostStreamTensorId(),
                              graph,
                              outTensors.at(0),
                              context);
  snap::program::Copy tmp_copy_prog(
      streamTensor, outTensors.at(0), false, context);
  prog.add(tmp_copy_prog);
}

snap::Tensor HostLoadDescriptorx::unwind(snap::Graph &graph,
                                         snap::Tensor tensor) const {
  poplar::DebugContext context;
  snap::Tensor unwound =
      makeWritableHostExchangeTensor(dv_p,
                                     TensorId(),
                                     descriptor.getHostStreamTensorId(),
                                     graph,
                                     tensor,
                                     context);
  return unwound;
}

void HostStoreDescriptorx::pre(snap::Graph &graph,
                               snap::program::Sequence &prog,
                               poplar::DebugContext context) {
  auto streamTensor =
      getOrCreateStreamTensor(dv_p,
                              descriptor.getHostStreamTensorId(),
                              graph,
                              inTensors.at(0).second,
                              context);

  snap::program::Copy tmp_copy_prog(
      inTensors.at(0).second, streamTensor, false, context);
  prog.add(tmp_copy_prog);
}

void HostStoreDescriptorx::exchange(snap::Graph &graph,
                                    snap::program::Sequence &prog,
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
    bool rearrangeOnHost = dv_p->lowering().doRearrangeOnHost(
        dv_p->ir().getTensor(descriptor.getHostStreamTensorId()));

    logging::opx::debug(
        "Found host stream in getFromHostStreams {} doRearrangeOnHost: {}",
        descriptor.getHostStreamTensorId(),
        rearrangeOnHost);
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

    snap::program::Copy copy_prog(
        streamTensor, stream, rearrangeOnHost, context);
    prog.add(copy_prog);

  } else {
    throw error("Stream for tensor {} not found",
                descriptor.getHostStreamTensorId());
  }
}

void HostStoreDescriptorx::post(snap::Graph &graph,
                                snap::program::Sequence &prog,
                                poplar::DebugContext context) {}

void RemoteLoadDescriptorx::pre(snap::Graph &graph,
                                snap::program::Sequence &prog,
                                poplar::DebugContext context) {}

void RemoteLoadDescriptorx::exchange(snap::Graph &graph,
                                     snap::program::Sequence &prog,
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

  if (offset.valid() && offset.numElements() > 0) {
    snap::program::Copy copy_prog(buffer.first, rbTensor, offset, context);
    prog.add(copy_prog);
  } else {
    snap::program::Copy copy_prog(buffer.first, rbTensor, context);
    prog.add(copy_prog);
  }
}

void RemoteLoadDescriptorx::post(snap::Graph &graph,
                                 snap::program::Sequence &prog,
                                 poplar::DebugContext context) {
  outTensors.push_back(
      makeWritableRemoteExchangeTensor(dv_p,
                                       inTensors.at(0).first,
                                       descriptor.getRemoteBufferId(),
                                       graph,
                                       inTensors.at(0).second,
                                       descriptor.isInplace()));

  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  snap::Tensor rbTensor = buffer.second.value();
  snap::program::Copy tmp_copy_prog(rbTensor, outTensors.at(0), false, context);
  prog.add(tmp_copy_prog);
}

snap::Tensor RemoteLoadDescriptorx::unwind(snap::Graph &graph,
                                           snap::Tensor tensor) const {
  auto context = DebugContext();
  snap::Tensor unwound =
      makeWritableRemoteExchangeTensor(dv_p,
                                       TensorId(),
                                       descriptor.getRemoteBufferId(),
                                       graph,
                                       tensor,
                                       descriptor.isInplace());
  return unwound;
}

void RemoteStoreDescriptorx::pre(snap::Graph &graph,
                                 snap::program::Sequence &prog,
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
  snap::program::Copy tmp_copy_prog(
      inTensors.at(0).second, rbTensor, false, context);
  prog.add(tmp_copy_prog);
}

void RemoteStoreDescriptorx::exchange(snap::Graph &graph,
                                      snap::program::Sequence &prog,
                                      poplar::DebugContext context) {
  snap::Tensor offset;
  if (inTensors.size() > 1) {
    offset = inTensors.at(1).second;
  }

  auto buffer =
      dv_p->lowering().getRemoteBuffer(descriptor.getRemoteBufferId());
  snap::Tensor rbTensor = buffer.second.value();
  if (offset.valid() && offset.numElements() > 0) {
    snap::program::Copy copy_prog(rbTensor, buffer.first, offset, context);
    prog.add(copy_prog);
  } else {
    snap::program::Copy copy_prog(rbTensor, buffer.first, context);
    prog.add(copy_prog);
  }
}

void RemoteStoreDescriptorx::post(snap::Graph &graph,
                                  snap::program::Sequence &prog,
                                  poplar::DebugContext context) {}

} // namespace popx
} // namespace popart
