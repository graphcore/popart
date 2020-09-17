// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/remote.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/remotesetup.hpp>

namespace popart {

std::size_t RemoteSetup::id() { return typeid(RemoteSetup).hash_code(); }

bool RemoteSetup::apply(Graph &graph) const {
  logging::debug("[RemoteSetup] Started.");

  Ir &ir                 = graph.getIr();
  int64_t remoteBufferId = 0;

  // Create remote buffer info for RemoteLoad/RemoteStore/RemoteExchange ops
  // with set buffer ID
  for (Op *op : ir.getAllOps()) {
    if (RemoteLoadOp *loadOp = dynamic_cast<RemoteLoadOp *>(op)) {
      auto allRemoteBufferIds = ir.getAllRemoteBufferInfos();
      RemoteBufferId id       = loadOp->getRemoteBufferId();
      if (id > -1 && allRemoteBufferIds.find(id) == allRemoteBufferIds.end()) {
        auto info = RemoteBufferInfo(
            loadOp->output->tensor(RemoteLoadOp::getLocalTensorOutIndex())
                ->info,
            1);
        ir.setRemoteBufferInfo(id, info);
        remoteBufferId = std::max(remoteBufferId, id + 1);
      }
    }
    if (RemoteStoreOp *storeOp = dynamic_cast<RemoteStoreOp *>(op)) {
      auto allRemoteBufferIds = ir.getAllRemoteBufferInfos();
      RemoteBufferId id       = storeOp->getRemoteBufferId();
      if (id > -1 && allRemoteBufferIds.find(id) == allRemoteBufferIds.end()) {
        auto info = RemoteBufferInfo(
            storeOp->input->tensor(RemoteStoreOp::getLocalTensorInIndex())
                ->info,
            1);
        ir.setRemoteBufferInfo(id, info);
        remoteBufferId = std::max(remoteBufferId, id + 1);
      }
    }
    if (RemoteExchangeOp *exchangeOp = dynamic_cast<RemoteExchangeOp *>(op)) {
      auto allRemoteBufferIds = ir.getAllRemoteBufferInfos();
      for (InIndex inIndex = 0;
           inIndex < exchangeOp->numLoads() + exchangeOp->numStores();
           ++inIndex) {
        RemoteBufferId id = exchangeOp->getRemoteBufferId(inIndex);
        if (id > -1 &&
            allRemoteBufferIds.find(id) == allRemoteBufferIds.end()) {
          auto info =
              RemoteBufferInfo(exchangeOp->input
                                   ->tensor(exchangeOp->numLoads() +
                                            exchangeOp->numStores() + inIndex)
                                   ->info,
                               1);
          ir.setRemoteBufferInfo(id, info);
          remoteBufferId = std::max(remoteBufferId, id + 1);
        }
      }
    }
  }

  // Mapping from each RemoteArg to it's final consumers
  std::map<TensorId, std::set<std::pair<Op *, InIndex>>> argOpMap;
  std::map<std::pair<Op *, InIndex>, std::set<TensorId>> opArgMap;
  std::map<TensorId, std::pair<RemoteBufferId, RemoteBufferIndex>> argBufferMap;

  // TODO: Support for Loop and ID ranges per arg tensor ID

  for (TensorId &tensor_id : graph.getTensors().getAllTensorIds()) {
    Tensor *tensor = graph.getTensors().get(tensor_id);
    if (tensor->isRemoteArgTensor()) {
      logging::transform::trace("[RemoteSetup] Resolving RemoteArg tensor {}",
                                tensor_id);
      std::vector<Tensor *> traceFront;
      traceFront.push_back(tensor);

      while (traceFront.size() > 0) {
        Tensor *front = traceFront.back();
        traceFront.pop_back();
        for (Op *consumer : front->consumers.getOps()) {
          // Only certain ops can be on the path between RemoteArg and
          // RemoteLoad/Store.
          if (consumer->opid == Onnx::CustomOperators::RemoteLoad ||
              consumer->opid == Onnx::CustomOperators::RemoteStore ||
              consumer->opid == Onnx::CustomOperators::RemoteExchange) {
            for (InIndex inIndex : consumer->input->indices(front)) {
              argOpMap[tensor_id].insert({consumer, inIndex});
              opArgMap[{consumer, inIndex}].insert(tensor_id);
            }
          } else if (consumer->opid == Onnx::CustomOperators::Call_1) {
            CallOp *call = dynamic_cast<CallOp *>(consumer);

            auto indices = consumer->input->indices(front);
            for (auto index : indices) {
              auto t_id = call->getCalledGraph().getInputId(index);
              auto t    = call->getCalledGraph().getTensors().get(t_id);
              traceFront.push_back(t);
            }
          } else {
            logging::warn("[RemoteSetup] Unsupported Op {} in path from"
                          "RemoteArg tensor {}.",
                          consumer->debugName(),
                          tensor->id);
          }
        }
      }
    }
  }

  std::map<int64_t, std::set<VGraphId>> remoteBufferVGIDs;

  for (auto &argOp : argOpMap) {
    if (argBufferMap.find(argOp.first) == argBufferMap.end()) {
      int64_t remoteBufferIndex = 0;

      // All remoteArg tensors in a group will refer to the same RemoteBufferId
      std::set<TensorId> group;
      TensorInfo tensorInfo;
      group.insert(argOp.first);

      std::vector<std::pair<Op *, InIndex>> front;
      front.insert(front.end(), argOp.second.begin(), argOp.second.end());
      while (front.size() > 0) {
        Op *op          = front.back().first;
        InIndex inIndex = front.back().second;
        front.pop_back();
        for (TensorId tensor_id : opArgMap.at({op, inIndex})) {
          if (group.find(tensor_id) == group.end()) {
            group.insert(tensor_id);
            front.insert(front.end(),
                         argOpMap.at(tensor_id).begin(),
                         argOpMap.at(tensor_id).end());
          }
        }
      }
      logging::trace("[RemoteSetup] RemoteArg group: {}", group);
      for (TensorId tensor_id : group) {
        argBufferMap[tensor_id] = {remoteBufferId, remoteBufferIndex};
        auto remoteArgTensor    = graph.getTensors().get(tensor_id);
        *static_cast<int *>(remoteArgTensor->tensorData()->data()) =
            static_cast<int>(remoteBufferIndex);
        logging::transform::trace(
            "[RemoteSetup] RemoteArg {} buffer: {} index: {}",
            tensor_id,
            remoteBufferId,
            remoteBufferIndex);
        for (auto opAndIndex : argOpMap[tensor_id]) {
          Op *op          = opAndIndex.first;
          InIndex inIndex = opAndIndex.second;
          if (RemoteStoreOp *rs = dynamic_cast<RemoteStoreOp *>(op)) {
            rs->setRemoteBufferId(remoteBufferId);
            remoteBufferVGIDs[remoteBufferId].insert(
                rs->hasVirtualGraphId() ? rs->getVirtualGraphId()
                                        : unusedVGraphId);
            logging::transform::trace(
                "[RemoteSetup] Op {} connected to remote buffer {}. "
                "Tensor info {}.",
                rs->debugName(),
                remoteBufferId,
                rs->inInfo(RemoteStoreOp::getLocalTensorInIndex()));
          }
          if (RemoteLoadOp *rl = dynamic_cast<RemoteLoadOp *>(op)) {
            rl->setRemoteBufferId(remoteBufferId);
            remoteBufferVGIDs[remoteBufferId].insert(
                rl->hasVirtualGraphId() ? rl->getVirtualGraphId()
                                        : unusedVGraphId);
            logging::transform::trace(
                "[RemoteSetup] Op {} connected to remote buffer {}. "
                "Tensor info {}.",
                rl->debugName(),
                remoteBufferId,
                rl->outInfo(RemoteLoadOp::getLocalTensorOutIndex()));
          }
          if (RemoteExchangeOp *re = dynamic_cast<RemoteExchangeOp *>(op)) {
            auto localInIndex = inIndex % (re->numLoads() + re->numStores());
            re->setRemoteBufferId(localInIndex, remoteBufferId);
            remoteBufferVGIDs[remoteBufferId].insert(
                re->getIntrospectionInVirtualGraphId(inIndex).first);
            logging::transform::trace(
                "[RemoteSetup] Op {} index {} connected to remote buffer {}. "
                "Tensor info {}.",
                re->debugName(),
                localInIndex,
                remoteBufferId,
                re->inInfo(localInIndex));
          }
        }
        remoteBufferIndex++;
      }

      auto info = RemoteBufferInfo(tensorInfo, remoteBufferIndex);
      ir.setRemoteBufferInfo(remoteBufferId, info);

      remoteBufferId++;
    }
  }

  for (auto &mapping : remoteBufferVGIDs) {
    if (mapping.second.size() > 1) {
      if (logging::transform::isEnabled(logging::Level::Trace)) {
        logging::transform::trace("[RemoteSetup] Remote buffer ID {} maps to "
                                  "multiple virtual graphs {} with:",
                                  mapping.first,
                                  mapping.second);
        for (auto &argBuffer : argBufferMap) {
          auto &tensor_id       = argBuffer.first;
          auto &remoteBufferId_ = argBuffer.second.first;
          if (remoteBufferId_ != mapping.first)
            continue;
          logging::transform::trace("[RemoteSetup]   Tensor arg {} with:",
                                    tensor_id);
          for (auto opAndIndex : argOpMap[tensor_id]) {
            Op *op          = opAndIndex.first;
            InIndex inIndex = opAndIndex.second;
            logging::transform::trace(
                "[RemoteSetup] Op {} phase {} vgid {} {}.",
                op->opid,
                op->hasExecutionPhase() ? op->getExecutionPhase()
                                        : unusedExecutionPhase,
                graph.getIr().virtualGraphsEnabled()
                    ? op->getIntrospectionInVirtualGraphId(inIndex).first
                    : -1,
                op->debugName());
          }
        }
      }
      throw error("[RemoteSetup] Remote buffer ID {} maps to multiple virtual "
                  "graphs {}.",
                  mapping.first,
                  mapping.second);
    }
  }

  // Remote (variable) tensors
  for (TensorId &tensor_id : graph.getTensors().getAllTensorIds()) {
    Tensor *tensor = graph.getTensors().get(tensor_id);
    if (tensor->tensorLocationInfo.isRemote()) {
      auto arg_tensor_id = getRemoteArgTensorId(tensor_id);
      auto info          = argBufferMap.find(arg_tensor_id);
      if (info != argBufferMap.end()) {
        tensor->tensorLocationInfo.setRemoteBufferInfo(info->second.first,
                                                       info->second.second);
        logging::transform::trace("Setting Tensor location info {} -> {}@{}",
                                  tensor_id,
                                  info->second.first,
                                  info->second.second);
      } else {
        // If the Tensor did not have an argBuffer value it must not have been a
        // required remote tensor for the graph. Remove it's protected "remote"
        // status so it can be pruned by ir::removeIsolated later.
        tensor->tensorLocationInfo.setRemote(false);
      }
    }
  }

  // Verify every Remote*Op has valid RemoteBufferIDs
  for (Op *op : ir.getAllOps()) {
    if (RemoteLoadOp *loadOp = dynamic_cast<RemoteLoadOp *>(op)) {
      if (loadOp->getRemoteBufferId() < 0) {
        throw error("Op {} has no valid remote buffer set.", op->debugName());
      }
    }
    if (RemoteStoreOp *storeOp = dynamic_cast<RemoteStoreOp *>(op)) {
      if (storeOp->getRemoteBufferId() < 0) {
        throw error("Op {} has no valid remote buffer set.", op->debugName());
      }
    }
    if (RemoteExchangeOp *exchangeOp = dynamic_cast<RemoteExchangeOp *>(op)) {
      for (InIndex inIndex = 0;
           inIndex < exchangeOp->numLoads() + exchangeOp->numStores();
           ++inIndex) {
        if (exchangeOp->getRemoteBufferId(inIndex) < 0) {
          throw error("Op {} index {} has no valid remote buffer set.",
                      op->debugName(),
                      inIndex);
        }
      }
    }
  }

  logging::debug("[RemoteSetup] Done.");
  return true;
}

namespace {
// RemoteSetup
bool init = Transform::registerTransform(new RemoteSetup());
} // namespace

} // namespace popart
