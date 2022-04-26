// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <snap/Graph.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/popx/exchangebundle.hpp>

namespace popart {
namespace popx {

ExchangeBundle::ExchangeBundle(const Ir &ir_) : ir(ir_) {
  for (auto op : ir_.getAllOps()) {
    if (ExchangeBaseOp *multiOp = dynamic_cast<ExchangeBaseOp *>(op)) {
      std::map<RemoteBufferId, std::set<ExchangeDirection>> bufferDirections;
      for (int i = 0; i < multiOp->getNumExchanges(); ++i) {
        auto descriptor = multiOp->getExchangeDescriptor(i);
        if (descriptor.isRemoteExchange()) {
          bufferDirections[descriptor.getRemoteBufferId()].insert(
              descriptor.getDirection());
        }
      }
      for (auto &bufferDirection : bufferDirections) {
        if (bufferDirection.second.size() > 1) {
          remoteBufferSeparateLoadStorePadsRequired[bufferDirection.first] |=
              true;
        } else {
          remoteBufferSeparateLoadStorePadsRequired[bufferDirection.first] |=
              false;
        }
      }
    }
  }
}

const std::string ExchangeBundle::getRemoteBufferName(RemoteBufferId id) {
  return "RB_" + std::to_string(id);
}

bool ExchangeBundle::hasRemoteBuffer(RemoteBufferId id) const {
  return remoteBuffers.find(id) != remoteBuffers.end();
}

const std::pair<snap::RemoteBuffer, std::vector<snap::Tensor>> &
ExchangeBundle::getRemoteBuffer(RemoteBufferId id) const {
  return remoteBuffers.at(id);
}

void ExchangeBundle::createRemoteBuffer(
    snap::Graph &graph,
    RemoteBufferId id,
    const std::vector<snap::Tensor> &tensors) {
  auto info    = ir.getRemoteBufferInfo(id);
  auto name    = getRemoteBufferName(id);
  auto type    = tensors.front().elementType();
  auto size    = tensors.front().numElements();
  auto repeats = info.repeats;

  logging::devicex::info(
      "Creating remote buffer {}, type {}, size {}, repeats {}",
      name,
      type,
      size,
      repeats);

  remoteBuffers.insert(
      {id,
       {graph.addRemoteBuffer(name, type, size, repeats, true, false),
        tensors}});
}

bool ExchangeBundle::hasStreamTensor(TensorId tid) const {
  return streamTensors.find(tid) != streamTensors.end();
}

snap::Tensor ExchangeBundle::getStreamTensor(TensorId tid) const {
  return streamTensors.at(tid);
}
void ExchangeBundle::setStreamTensor(TensorId tid, snap::Tensor tensors) {
  streamTensors[tid] = tensors;
}

} // namespace popx
} // namespace popart
