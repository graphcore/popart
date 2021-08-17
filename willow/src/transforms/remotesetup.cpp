// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/dynamic/dynamicupdate.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/remotesetup.hpp>

namespace popart {

void RemoteSetup::getRemoteArgMapping(Graph &graph,
                                      RemoteArgOpMap &argOpMap,
                                      RemoteOpArgMap &opArgMap,
                                      RemoteArgBufferMap &argBufferMap) {
  for (TensorId &tensor_id : graph.getTensors().getAllTensorIds()) {
    Tensor *tensor = graph.getTensors().get(tensor_id);
    if (tensor->isRemoteArgTensor() &&
        tensor->tensorType() == TensorType::Const) {
      logging::transform::trace("[RemoteSetup] Resolving RemoteArg tensor {}",
                                tensor_id);
      std::vector<std::pair<Tensor *, std::vector<SubgraphOp *>>> traceFront;
      traceFront.push_back({tensor, {}});

      std::set<std::pair<Tensor *, std::vector<SubgraphOp *>>> visited;
      while (traceFront.size() > 0) {
        auto front     = traceFront.back();
        Tensor *tfront = front.first;
        auto callStack = front.second;
        traceFront.pop_back();

        if (visited.find(front) != visited.end()) {
          continue;
        }
        visited.insert(front);

        if (tfront->isGraphOutput()) {
          auto nextStack   = callStack;
          SubgraphOp *sgOp = nextStack.back();
          nextStack.pop_back();
          auto sgOutIndex = sgOp->getCalledGraph().getOutputIndex(tfront->id);
          auto opOutIndex = sgOp->subgraphOutToOpOutIndex(sgOutIndex);
          traceFront.push_back({sgOp->outTensor(opOutIndex), nextStack});
          // Check for loop carried tensors
          if (sgOp->isConvertibleTo<LoopOp>()) {
            // See loop.hpp, output m maps to input m+1
            InIndex sgInIndex = sgOutIndex + 1;
            traceFront.push_back(
                {sgOp->getCalledGraph().getTensors().get(
                     sgOp->getCalledGraph().getInputId(sgInIndex)),
                 callStack});
          }
        }

        for (Op *consumer : tfront->consumers.getOps()) {
          // Only certain ops can be on the path between RemoteArg and
          // RemoteLoad/Store.
          if (consumer->opid == Onnx::CustomOperators::RemoteLoad ||
              consumer->opid == Onnx::CustomOperators::RemoteStore ||
              consumer->opid == Onnx::CustomOperators::MultiExchange) {
            for (InIndex inIndex : consumer->input->indices(tfront)) {
              argOpMap[tensor_id].insert({consumer, inIndex});
              opArgMap[{consumer, inIndex}].insert(tensor_id);
            }
          } else if (consumer->isConvertibleTo<SubgraphOp>()) {
            SubgraphOp *subgraphOp = dynamic_cast<SubgraphOp *>(consumer);
            auto indices           = consumer->input->indices(tfront);
            for (auto index : indices) {
              auto sgIndex = subgraphOp->opInToSubgraphInIndex(index);
              if (sgIndex > -1 &&
                  sgIndex < subgraphOp->getCalledGraph().getInputIds().size()) {
                auto t_id = subgraphOp->getCalledGraph().getInputId(sgIndex);
                auto t    = subgraphOp->getCalledGraph().getTensors().get(t_id);
                auto nextCallStack = callStack;
                nextCallStack.push_back(subgraphOp);
                traceFront.push_back({t, nextCallStack});
              }
            }
          } else if (consumer->isConvertibleTo<ElementWiseBinaryBaseOp>()) {
            traceFront.push_back({consumer->output->tensor(
                                      ElementWiseBinaryBaseOp::getOutIndex()),
                                  callStack});
          } else if (consumer->isConvertibleTo<ConcatOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(ConcatOp::getOutIndex()), callStack});
          } else if (consumer->isConvertibleTo<SliceOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(SliceOp::getOutIndex()), callStack});
          } else if (consumer->isConvertibleTo<ConcatInplaceOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(ConcatInplaceOp::getOutIndex()),
                 callStack});
          } else if (consumer->isConvertibleTo<SliceInplaceOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(SliceInplaceOp::getOutIndex()),
                 callStack});
          } else if (consumer->isConvertibleTo<DynamicUpdateOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(DynamicUpdateOp::getOutIndex()),
                 callStack});
          } else if (consumer->isConvertibleTo<DynamicUpdateInplaceOp>()) {
            traceFront.push_back({consumer->output->tensor(
                                      DynamicUpdateInplaceOp::getOutIndex()),
                                  callStack});
          } else if (consumer->isConvertibleTo<DynamicSliceOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(DynamicSliceOp::getOutIndex()),
                 callStack});
          } else if (consumer->isConvertibleTo<IdentityOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(IdentityOp::getOutIndex()),
                 callStack});
          } else if (consumer->isConvertibleTo<ReshapeOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(ReshapeOp::getOutIndex()),
                 callStack});
          } else if (consumer->isConvertibleTo<ReshapeInplaceOp>()) {
            traceFront.push_back(
                {consumer->output->tensor(ReshapeInplaceOp::getOutIndex()),
                 callStack});
          } else {
            logging::debug("[RemoteSetup] Unsupported Op {} in path from"
                           "RemoteArg tensor {}.",
                           consumer->debugName(),
                           tensor->id);
          }
        }
      }
    }
  }
}

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
    if (MultiExchangeOp *exchangeOp = dynamic_cast<MultiExchangeOp *>(op)) {
      auto allRemoteBufferIds = ir.getAllRemoteBufferInfos();

      for (int index = 0; index < exchangeOp->getNumExchanges(); ++index) {
        if (exchangeOp->isRemote(index)) {
          RemoteBufferId id = exchangeOp->getRemoteBufferId(index);
          if (id > -1 &&
              allRemoteBufferIds.find(id) == allRemoteBufferIds.end()) {
            auto info = RemoteBufferInfo(
                exchangeOp->input
                    ->tensor(
                        exchangeOp->descriptorIndexToInIndices(index).at(0))
                    ->info,
                1);
            ir.setRemoteBufferInfo(id, info);
            remoteBufferId = std::max(remoteBufferId, id + 1);
          }
        }
      }
    }
  }

  // Mapping from each RemoteArg to it's final consumers
  RemoteArgOpMap argOpMap;
  RemoteOpArgMap opArgMap;
  RemoteArgBufferMap argBufferMap;

  getRemoteArgMapping(graph, argOpMap, opArgMap, argBufferMap);

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
      for (TensorId remoteArgId : group) {
        argBufferMap[remoteArgId] = {remoteBufferId, remoteBufferIndex};
        auto remoteArgTensor      = graph.getTensors().get(remoteArgId);
        *static_cast<int *>(remoteArgTensor->tensorData()->data()) =
            static_cast<int>(remoteBufferIndex);
        logging::transform::trace(
            "[RemoteSetup] RemoteArg {} buffer: {} index: {}",
            remoteArgId,
            remoteBufferId,
            remoteBufferIndex);
        for (auto opAndIndex : argOpMap[remoteArgId]) {
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
          if (MultiExchangeOp *exchangeOp =
                  dynamic_cast<MultiExchangeOp *>(op)) {
            auto indices = exchangeOp->inIndexToDescriptorIndex(inIndex);
            if (exchangeOp->isRemote(indices.first)) {
              exchangeOp->setRemoteBufferId(indices.first, remoteBufferId);
            }
            InIndex localInIndex = inIndex - 1;
            remoteBufferVGIDs[remoteBufferId].insert(
                exchangeOp->Op::getIntrospectionInVirtualGraphId(localInIndex)
                    .first);
            logging::transform::trace(
                "[RemoteSetup] Op {} index {} connected to remote buffer {}. "
                "Tensor info {}.",
                exchangeOp->debugName(),
                localInIndex,
                remoteBufferId,
                exchangeOp->inInfo(localInIndex));
          }
        }
        // Remote arg is a single index: [index]
        // increment the remoteBufferIndex by 1
        int64_t inc = 1;
        if (remoteArgTensor->info.shape() == Shape{2}) {
          // Remote arg is a range of indices: [start, size]
          // increment the remoteBufferIndex by size
          inc =
              *(static_cast<int *>(remoteArgTensor->tensorData()->data()) + 1);
        }
        logging::transform::trace("[RemoteSetup] incrementing {} {} by {}",
                                  remoteArgId,
                                  remoteBufferIndex,
                                  inc);
        remoteBufferIndex += inc;
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
                    : unusedVGraphId,
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
    if (MultiExchangeOp *exchangeOp = dynamic_cast<MultiExchangeOp *>(op)) {
      for (int index = 0; index < exchangeOp->getNumExchanges(); ++index) {
        if (exchangeOp->getExchangeDescriptor(index).isRemoteExchange() &&
            exchangeOp->getRemoteBufferId(index) < 0) {
          throw error(
              "Op {} descriptor index {} has no valid remote buffer set.",
              op->debugName(),
              index);
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
