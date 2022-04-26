// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <map>
#include <memory>
#include <snap/DataStream.hpp>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <utility>
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplar/TensorCloneMethod.hpp>
#include <popops/HostSliceTensor.hpp>
#include <popart/ir.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/exchange/exchangex.hpp>

#include "popart/debugcontext.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op/exchange/exchange.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
class Op;

namespace popx {

namespace {

std::vector<snap::Tensor>
createRemoteBufferLandingPads(snap::Graph &graph,
                              snap::Tensor refTensor,
                              ExchangeDescriptor descriptor,
                              bool remoteBufferSeparateLoadStorePadsRequired) {
  auto numPads = remoteBufferSeparateLoadStorePadsRequired ? 2 : 1;
  std::vector<snap::Tensor> rbTensors;
  rbTensors.reserve(numPads);
  for (int i = 0; i < numPads; ++i) {
    rbTensors.push_back(
        graph.clone(refTensor,
                    poplar::DebugContext(ExchangeBundle::getRemoteBufferName(
                        descriptor.getRemoteBufferId())),
                    poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES));
  }
  return rbTensors;
}

snap::Tensor selectRemoteBufferLandingPad(std::vector<snap::Tensor> rbTensors,
                                          ExchangeDescriptor descriptor) {
  return rbTensors.at(static_cast<int>(descriptor.getDirection()) %
                      rbTensors.size());
}

snap::Tensor makeWritableRemoteExchangeTensor(Devicex *dv_p,
                                              ExchangeDescriptor descriptor,
                                              TensorId id,
                                              snap::Graph &graph,
                                              snap::Tensor t) {
  auto rbid    = descriptor.getRemoteBufferId();
  bool inplace = descriptor.isInplace();
  std::vector<snap::Tensor> rbTensors;
  // Clone the input tensor if the remote buffer tensor does not exist
  if (!dv_p->lowering().getExchangeBundle().hasRemoteBuffer(rbid)) {
    rbTensors = createRemoteBufferLandingPads(
        graph,
        t,
        descriptor,
        dv_p->lowering()
            .getExchangeBundle()
            .getRemoteBufferSeparateLoadStorePadsRequired(rbid));
    dv_p->lowering().getExchangeBundle().createRemoteBuffer(
        graph, rbid, rbTensors);
  }
  auto buffer = dv_p->lowering().getExchangeBundle().getRemoteBuffer(rbid);
  rbTensors   = buffer.second;
  // The outplacing of the tensor will be done explicitly
  if (!inplace) {
    snap::Tensor tw =
        graph.clone(selectRemoteBufferLandingPad(rbTensors, descriptor),
                    {id + "_writable"},
                    poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    return tw;
  }
  // Outplace fallback
  if (!t.isParallelWriteable() || t.getPoplarTensor().containsConstant()) {
    logging::opx::warn("Tensor {} is not a writable remote buffer "
                       "copy target, cloning. "
                       "The aliasing properties have changed implicitly.",
                       id);
    snap::Tensor tw =
        graph.clone(selectRemoteBufferLandingPad(rbTensors, descriptor),
                    {id + "_writable"},
                    poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
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

  if (dv_p->lowering().getExchangeBundle().hasStreamTensor("ST_" +
                                                           streamTensorId)) {
    streamTensor = dv_p->lowering().getExchangeBundle().getStreamTensor(
        "ST_" + streamTensorId);
  } else {
    streamTensor =
        graph.clone(t,
                    poplar::DebugContext(context, streamTensorId),
                    poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().getExchangeBundle().setStreamTensor("ST_" + streamTensorId,
                                                         streamTensor);
  }
  return streamTensor;
}

snap::Tensor
makeWritableHostExchangeTensor(Devicex *dv_p,
                               TensorId id,
                               TensorId streamTensorId,
                               snap::Graph &graph,
                               snap::Tensor t,
                               bool inplace,
                               const poplar::DebugContext &context) {
  snap::Tensor streamTensor =
      getOrCreateStreamTensor(dv_p, id, graph, t, context);
  if (dv_p->lowering().tensors().contains(streamTensorId)) {
    streamTensor = dv_p->lowering().tensors().get(streamTensorId);
  } else {
    streamTensor =
        graph.clone(t,
                    {streamTensorId},
                    poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().getExchangeBundle().setStreamTensor(id, streamTensor);
  }
  // The outplacing of the tensor will be done explicitly
  if (!inplace) {
    snap::Tensor tw =
        graph.clone(streamTensor,
                    {id + "_writable"},
                    poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    return tw;
  }
  // Outplace fallback
  if (!t.isParallelWriteable()) {
    logging::opx::debug("Tensor {} is not a writable host load tensor "
                        " target, cloning. "
                        "The aliasing properties have changed implicitly.",
                        id);
    snap::Tensor tw =
        graph.clone(streamTensor,
                    {id + "_writable"},
                    poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
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

snap::Tensor ExchangeDescriptorx::create(snap::Graph &graph,
                                         const TensorInfo &info) const {
  std::string debugContext =
      descriptor.isHostExchange()
          ? descriptor.getHostStreamTensorId()
          : std::to_string(descriptor.getRemoteBufferId());

  // Note: ExchangeDirection::Store means isRead is true for the host side
  return snap::Tensor{popops::createHostTransferableTensor(
                          graph.getPoplarGraph(),
                          popType(info.getDataTypeInfo()->type()),
                          info.shape_szt(),
                          descriptor.getDirection() == ExchangeDirection::Store,
                          {debugContext}),
                      graph};
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
    logging::opx::debug(
        "Found host stream in getFromHostStreams {} doRearrangeOnHost: {}",
        descriptor.getHostStreamTensorId(),
        rearrangeOnHost());
    auto stream = streams.at(descriptor.getHostStreamTensorId());

    snap::program::Copy copy_prog(
        stream, streamTensor, rearrangeOnHost(), context);
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
                                     descriptor.isInplace(),
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
                                     descriptor.isInplace(),
                                     context);
  return unwound;
}

bool HostLoadDescriptorx::rearrangeOnHost() const {
  return dv_p->lowering().doRearrangeOnHost(
      dv_p->ir().getTensor(descriptor.getHostStreamTensorId()));
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

bool HostStoreDescriptorx::rearrangeOnHost() const {
  return dv_p->lowering().doRearrangeOnHost(
      dv_p->ir().getTensor(descriptor.getHostStreamTensorId()));
}

void RemoteLoadDescriptorx::pre(snap::Graph &graph,
                                snap::program::Sequence &prog,
                                poplar::DebugContext context) {}

void RemoteLoadDescriptorx::exchange(snap::Graph &graph,
                                     snap::program::Sequence &prog,
                                     poplar::DebugContext context) {
  std::vector<snap::Tensor> rbTensors;

  snap::Tensor offset;
  if (inTensors.size() > 1) {
    offset = inTensors.at(1).second;
  }

  if (!dv_p->lowering().getExchangeBundle().hasRemoteBuffer(
          descriptor.getRemoteBufferId())) {
    auto rbTensors = createRemoteBufferLandingPads(
        graph,
        inTensors.at(0).second,
        descriptor,
        dv_p->lowering()
            .getExchangeBundle()
            .getRemoteBufferSeparateLoadStorePadsRequired(
                descriptor.getRemoteBufferId()));
    dv_p->lowering().getExchangeBundle().createRemoteBuffer(
        graph, descriptor.getRemoteBufferId(), rbTensors);
  }

  auto buffer = dv_p->lowering().getExchangeBundle().getRemoteBuffer(
      descriptor.getRemoteBufferId());
  rbTensors = buffer.second;

  if (offset.valid() && offset.numElements() > 0) {
    snap::program::Copy copy_prog(
        buffer.first,
        selectRemoteBufferLandingPad(rbTensors, descriptor),
        offset,
        context);
    prog.add(copy_prog);
  } else {
    snap::program::Copy copy_prog(
        buffer.first,
        selectRemoteBufferLandingPad(rbTensors, descriptor),
        context);
    prog.add(copy_prog);
  }
}

void RemoteLoadDescriptorx::post(snap::Graph &graph,
                                 snap::program::Sequence &prog,
                                 poplar::DebugContext context) {
  outTensors.push_back(makeWritableRemoteExchangeTensor(
      dv_p, descriptor, inTensors.at(0).first, graph, inTensors.at(0).second));

  auto buffer = dv_p->lowering().getExchangeBundle().getRemoteBuffer(
      descriptor.getRemoteBufferId());
  std::vector<snap::Tensor> rbTensors = buffer.second;
  snap::program::Copy tmp_copy_prog(
      selectRemoteBufferLandingPad(rbTensors, descriptor),
      outTensors.at(0),
      false,
      context);
  prog.add(tmp_copy_prog);
}

snap::Tensor RemoteLoadDescriptorx::unwind(snap::Graph &graph,
                                           snap::Tensor tensor) const {
  auto context         = DebugContext();
  snap::Tensor unwound = makeWritableRemoteExchangeTensor(
      dv_p, descriptor, TensorId(), graph, tensor);
  return unwound;
}

void RemoteStoreDescriptorx::pre(snap::Graph &graph,
                                 snap::program::Sequence &prog,
                                 poplar::DebugContext context) {
  std::vector<snap::Tensor> rbTensors;
  if (!dv_p->lowering().getExchangeBundle().hasRemoteBuffer(
          descriptor.getRemoteBufferId())) {
    rbTensors = createRemoteBufferLandingPads(
        graph,
        inTensors.at(0).second,
        descriptor,
        dv_p->lowering()
            .getExchangeBundle()
            .getRemoteBufferSeparateLoadStorePadsRequired(
                descriptor.getRemoteBufferId()));
    dv_p->lowering().getExchangeBundle().createRemoteBuffer(
        graph, descriptor.getRemoteBufferId(), rbTensors);
  }
  auto buffer = dv_p->lowering().getExchangeBundle().getRemoteBuffer(
      descriptor.getRemoteBufferId());
  rbTensors = buffer.second;
  snap::program::Copy tmp_copy_prog(
      inTensors.at(0).second,
      selectRemoteBufferLandingPad(rbTensors, descriptor),
      false,
      context);
  prog.add(tmp_copy_prog);
}

void RemoteStoreDescriptorx::exchange(snap::Graph &graph,
                                      snap::program::Sequence &prog,
                                      poplar::DebugContext context) {
  snap::Tensor offset;
  if (inTensors.size() > 1) {
    offset = inTensors.at(1).second;
  }

  auto buffer = dv_p->lowering().getExchangeBundle().getRemoteBuffer(
      descriptor.getRemoteBufferId());
  std::vector<snap::Tensor> rbTensors = buffer.second;
  if (offset.valid() && offset.numElements() > 0) {
    snap::program::Copy copy_prog(
        selectRemoteBufferLandingPad(rbTensors, descriptor),
        buffer.first,
        offset,
        context);
    prog.add(copy_prog);
  } else {
    snap::program::Copy copy_prog(
        selectRemoteBufferLandingPad(rbTensors, descriptor),
        buffer.first,
        context);
    prog.add(copy_prog);
  }
}

void RemoteStoreDescriptorx::post(snap::Graph &graph,
                                  snap::program::Sequence &prog,
                                  poplar::DebugContext context) {}

} // namespace popx
} // namespace popart
