// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <poplar/DataStream.hpp>
#include <poplar/FunctionBufferMappingType.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/TensorCloneMethod.hpp>
#include <popops/HostSliceTensor.hpp>
#include <popart/ir.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/exchange/exchangex.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/debugcontext.hpp"
#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/op/exchange/exchange.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/exchangebundle.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
class Op;

namespace popx {

namespace {

std::vector<poplar::Tensor>
createRemoteBufferLandingPads(poplar::Graph &graph,
                              poplar::Tensor refTensor,
                              ExchangeDescriptor descriptor,
                              bool remoteBufferSeparateLoadStorePadsRequired) {
  auto numPads = remoteBufferSeparateLoadStorePadsRequired ? 2 : 1;
  std::vector<poplar::Tensor> rbTensors;
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

poplar::Tensor
selectRemoteBufferLandingPad(std::vector<poplar::Tensor> rbTensors,
                             ExchangeDescriptor descriptor) {
  return rbTensors.at(static_cast<int>(descriptor.getDirection()) %
                      rbTensors.size());
}

poplar::Tensor makeWritableRemoteExchangeTensor(Devicex *dv_p,
                                                ExchangeDescriptor descriptor,
                                                TensorId id,
                                                poplar::Graph &graph,
                                                poplar::Tensor t) {
  auto rbid    = descriptor.getRemoteBufferId();
  bool inplace = descriptor.isInplace();
  std::vector<poplar::Tensor> rbTensors;
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
    poplar::Tensor tw =
        graph.clone(selectRemoteBufferLandingPad(rbTensors, descriptor),
                    {id + "_writable"},
                    poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    return tw;
  }
  // Outplace fallback
  if (!t.isParallelWriteable() || t.containsConstant()) {
    logging::opx::warn("Tensor {} is not a writable remote buffer "
                       "copy target, cloning. "
                       "The aliasing properties have changed implicitly.",
                       id);
    poplar::Tensor tw =
        graph.clone(selectRemoteBufferLandingPad(rbTensors, descriptor),
                    {id + "_writable"},
                    poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    return tw;
  }
  return t;
}

poplar::Tensor getOrCreateStreamTensor(Devicex *dv_p,
                                       TensorId streamTensorId,
                                       poplar::Graph &graph,
                                       poplar::Tensor t,
                                       const poplar::DebugContext &context) {
  poplar::Tensor streamTensor;

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

poplar::Tensor
makeWritableHostExchangeTensor(Devicex *dv_p,
                               TensorId id,
                               TensorId streamTensorId,
                               poplar::Graph &graph,
                               poplar::Tensor t,
                               bool inplace,
                               const poplar::DebugContext &context) {
  poplar::Tensor streamTensor =
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
    poplar::Tensor tw =
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
    poplar::Tensor tw =
        graph.clone(streamTensor,
                    {id + "_writable"},
                    poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    return tw;
  }
  return t;
}

} // namespace

ExchangeBaseOpx::ExchangeBaseOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

ExchangeDescriptorx::ExchangeDescriptorx(Devicex *dv_p_,
                                         ExchangeDescriptor descriptor_)
    : dv_p(dv_p_), descriptor(descriptor_) {}

poplar::Tensor ExchangeDescriptorx::unwind(poplar::Graph &,
                                           poplar::Tensor tensor) const {
  return tensor;
}

poplar::Tensor ExchangeDescriptorx::create(poplar::Graph &graph,
                                           const TensorInfo &info) const {
  std::string debugContext =
      descriptor.isHostExchange()
          ? descriptor.getHostStreamTensorId()
          : std::to_string(descriptor.getRemoteBufferId());

  poplar::Tensor t;
  auto &lowering = dv_p->lowering();
  auto withOffset =
      lowering.ir()
          .getSessionOptions()
          .experimentalSettings.createHostTransferableTensorWithOffset;

  // Note: ExchangeDirection::Store means isRead is true for the host side
  if (!withOffset) {
    t = popops::createHostTransferableTensor(
        graph,
        popType(info.getDataTypeInfo()->type()),
        info.shape_szt(),
        descriptor.getDirection() == ExchangeDirection::Store,
        {debugContext});
  } else {
    auto &offsetMap = lowering.getInitTensorOffsetMap();
    auto offset     = offsetMap.getOffset(graph);
    t               = popops::createHostTransferableTensor(
        graph,
        popType(info.getDataTypeInfo()->type()),
        info.shape_szt(),
        descriptor.getDirection() == ExchangeDirection::Store,
        offset,
        {debugContext});
    auto dtype = popType(info.getDataTypeInfo()->type());
    offset += graph.getTarget().getTypeSize(dtype) * t.numElements();
    offsetMap.setOffset(graph, offset);
  }

  return t;
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
  } else if (descriptor.isCodeCopy()) {
    switch (descriptor.getDirection()) {
    case ExchangeDirection::Load: {
      return std::make_unique<RemoteCodeLoadOpDescriptorx>(dv_p, descriptor);
      break;
    }
    case ExchangeDirection::Store: {
      throw error(
          "ExchangeDirection::Store not support for RemoteCodeLoadOp op.");
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

RemoteCodeLoadOpDescriptorx::RemoteCodeLoadOpDescriptorx(
    Devicex *dv_p_,
    ExchangeDescriptor descriptor_)
    : ExchangeDescriptorx(dv_p_, descriptor_) {}

void HostLoadDescriptorx::pre(poplar::Graph &graph,
                              poplar::program::Sequence &prog,
                              poplar::DebugContext context) {}

void HostLoadDescriptorx::exchange(poplar::Graph &graph,
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
    logging::opx::debug(
        "Found host stream in getFromHostStreams {} doRearrangeOnHost: {}",
        descriptor.getHostStreamTensorId(),
        rearrangeOnHost());
    auto stream = streams.at(descriptor.getHostStreamTensorId());

    poplar::program::Copy copy_prog(
        stream, streamTensor, rearrangeOnHost(), context);
    prog.add(copy_prog);

  } else {
    throw error("Stream for tensor {} not found",
                descriptor.getHostStreamTensorId());
  }
}

void HostLoadDescriptorx::post(poplar::Graph &graph,
                               poplar::program::Sequence &prog,
                               poplar::DebugContext context) {
  outTensors.push_back(
      makeWritableHostExchangeTensor(dv_p,
                                     inTensors.at(0).first,
                                     descriptor.getHostStreamTensorId(),
                                     graph,
                                     inTensors.at(0).second,
                                     descriptor.isInplace(),
                                     context));

  poplar::Tensor streamTensor =
      getOrCreateStreamTensor(dv_p,
                              descriptor.getHostStreamTensorId(),
                              graph,
                              outTensors.at(0),
                              context);
  poplar::program::Copy tmp_copy_prog(
      streamTensor, outTensors.at(0), false, context);
  prog.add(tmp_copy_prog);
}

poplar::Tensor HostLoadDescriptorx::unwind(poplar::Graph &graph,
                                           poplar::Tensor tensor) const {
  poplar::DebugContext context;
  poplar::Tensor unwound =
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

void HostStoreDescriptorx::pre(poplar::Graph &graph,
                               poplar::program::Sequence &prog,
                               poplar::DebugContext context) {
  auto streamTensor =
      getOrCreateStreamTensor(dv_p,
                              descriptor.getHostStreamTensorId(),
                              graph,
                              inTensors.at(0).second,
                              context);

  poplar::program::Copy tmp_copy_prog(
      inTensors.at(0).second, streamTensor, false, context);
  prog.add(tmp_copy_prog);
}

void HostStoreDescriptorx::exchange(poplar::Graph &graph,
                                    poplar::program::Sequence &prog,
                                    poplar::DebugContext context) {
  auto streams = dv_p->lowering().getToHostAnchorStreams();
  auto it      = streams.find(descriptor.getHostStreamTensorId());

  poplar::Tensor streamTensor =
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

    poplar::program::Copy copy_prog(
        streamTensor, stream, rearrangeOnHost, context);
    prog.add(copy_prog);

  } else {
    throw error("Stream for tensor {} not found",
                descriptor.getHostStreamTensorId());
  }
}

void HostStoreDescriptorx::post(poplar::Graph &graph,
                                poplar::program::Sequence &prog,
                                poplar::DebugContext context) {}

bool HostStoreDescriptorx::rearrangeOnHost() const {
  return dv_p->lowering().doRearrangeOnHost(
      dv_p->ir().getTensor(descriptor.getHostStreamTensorId()));
}

void RemoteLoadDescriptorx::pre(poplar::Graph &graph,
                                poplar::program::Sequence &prog,
                                poplar::DebugContext context) {}

void RemoteLoadDescriptorx::exchange(poplar::Graph &graph,
                                     poplar::program::Sequence &prog,
                                     poplar::DebugContext context) {
  std::vector<poplar::Tensor> rbTensors;

  poplar::Tensor offset;
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
    poplar::program::Copy copy_prog(
        buffer.first,
        selectRemoteBufferLandingPad(rbTensors, descriptor),
        offset,
        context);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(
        buffer.first,
        selectRemoteBufferLandingPad(rbTensors, descriptor),
        context);
    prog.add(copy_prog);
  }
}

void RemoteLoadDescriptorx::post(poplar::Graph &graph,
                                 poplar::program::Sequence &prog,
                                 poplar::DebugContext context) {
  outTensors.push_back(makeWritableRemoteExchangeTensor(
      dv_p, descriptor, inTensors.at(0).first, graph, inTensors.at(0).second));

  auto buffer = dv_p->lowering().getExchangeBundle().getRemoteBuffer(
      descriptor.getRemoteBufferId());
  std::vector<poplar::Tensor> rbTensors = buffer.second;
  poplar::program::Copy tmp_copy_prog(
      selectRemoteBufferLandingPad(rbTensors, descriptor),
      outTensors.at(0),
      false,
      context);
  prog.add(tmp_copy_prog);
}

poplar::Tensor RemoteLoadDescriptorx::unwind(poplar::Graph &graph,
                                             poplar::Tensor tensor) const {
  auto context           = DebugContext();
  poplar::Tensor unwound = makeWritableRemoteExchangeTensor(
      dv_p, descriptor, TensorId(), graph, tensor);
  return unwound;
}

void RemoteStoreDescriptorx::pre(poplar::Graph &graph,
                                 poplar::program::Sequence &prog,
                                 poplar::DebugContext context) {
  std::vector<poplar::Tensor> rbTensors;
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
  poplar::program::Copy tmp_copy_prog(
      inTensors.at(0).second,
      selectRemoteBufferLandingPad(rbTensors, descriptor),
      false,
      context);
  prog.add(tmp_copy_prog);
}

void RemoteStoreDescriptorx::exchange(poplar::Graph &graph,
                                      poplar::program::Sequence &prog,
                                      poplar::DebugContext context) {
  poplar::Tensor offset;
  if (inTensors.size() > 1) {
    offset = inTensors.at(1).second;
  }

  auto buffer = dv_p->lowering().getExchangeBundle().getRemoteBuffer(
      descriptor.getRemoteBufferId());
  std::vector<poplar::Tensor> rbTensors = buffer.second;
  if (offset.valid() && offset.numElements() > 0) {
    poplar::program::Copy copy_prog(
        selectRemoteBufferLandingPad(rbTensors, descriptor),
        buffer.first,
        offset,
        context);
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(
        selectRemoteBufferLandingPad(rbTensors, descriptor),
        buffer.first,
        context);
    prog.add(copy_prog);
  }
}

void RemoteStoreDescriptorx::post(poplar::Graph &graph,
                                  poplar::program::Sequence &prog,
                                  poplar::DebugContext context) {}

void RemoteCodeLoadOpDescriptorx::pre(poplar::Graph &graph,
                                      poplar::program::Sequence &prog,
                                      poplar::DebugContext context) {
  auto fbmt =
      getFunctionBufferMappingType(*descriptor.getDestinationCodeMemoryType());
  dv_p->lowering().addFunctionBuffers(*descriptor.getGraphToLoadId(), fbmt);
}

void RemoteCodeLoadOpDescriptorx::exchange(poplar::Graph &graph,
                                           poplar::program::Sequence &prog,
                                           poplar::DebugContext context) {
  auto gid = *descriptor.getGraphToLoadId();
  auto fbmt =
      getFunctionBufferMappingType(*descriptor.getDestinationCodeMemoryType());
  if (dv_p->lowering().hasFunctionBuffer(gid, fbmt)) {
    auto graph_progs = dv_p->lowering().getFunctionBuffer(gid, fbmt);

    for (auto pair : graph_progs) {
      auto f      = pair.first;
      auto buffer = pair.second;

      prog.add(poplar::program::Copy(buffer, f, context));
    }

  } else {
    throw error("No poplar::FunctionBuffer found for graph id {}", gid.str());
  }
}

void RemoteCodeLoadOpDescriptorx::post(poplar::Graph &graph,
                                       poplar::program::Sequence &prog,
                                       poplar::DebugContext context) {}

poplar::FunctionBufferMappingType
RemoteCodeLoadOpDescriptorx::getFunctionBufferMappingType(
    CodeMemoryType destination) {
  switch (destination) {
  case popart::CodeMemoryType::Buffer:;
    throw error(
        "LocationType `Buffer` not yet supported for RemoteCodeLoadOpOp");
  case popart::CodeMemoryType::ExecutableMemory:
    return poplar::FunctionBufferMappingType::REMOTE;
  case popart::CodeMemoryType::N:
  default:
    break;
  }
  throw error("Unsupported LocationType for RemoteCodeLoadOpOp");
}

} // namespace popx
} // namespace popart
