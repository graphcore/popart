// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <transforms/streamingmemoryopinserter.hpp>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/tensor.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/iocomputetilecopy.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/transforms/transform.hpp"

namespace popart {

std::size_t IoComputeTileCopy::id() {
  return typeid(IoComputeTileCopy).hash_code();
}

TensorId IoComputeTileCopy::generateCopiedTensorId(Tensor *tensor,
                                                   TileSet toIoTiles) const {
  TensorId copiedTensor =
      tensor->id + (toIoTiles == TileSet::IO ? "_tioc" : "_fioc");
  return copiedTensor;
}

void IoComputeTileCopy::connectIoTileCopy(Graph &,
                                          Tensor *tensor,
                                          TileSet toTileSet,
                                          Op *toOp,
                                          InIndex inIndex) const {

  // Remove this input tensor from the to op for each index
  logging::transform::debug("Disconnecting out {} from {}:{}",
                            tensor->id,
                            toOp->debugName(),
                            inIndex);
  toOp->disconnectInTensor(inIndex, tensor);

  TensorId copiedTensor = generateCopiedTensorId(tensor, toTileSet);

  // Add the copied input tensor to the to op for each index
  logging::transform::debug(
      "Connecting in {} from {}:{}", copiedTensor, toOp->debugName(), inIndex);
  toOp->connectInTensor(inIndex, copiedTensor);
}

void IoComputeTileCopy::insertIoTileCopy(Graph &graph,
                                         Tensor *tensor,
                                         TileSet fromTileSet,
                                         TileSet toTileSet,
                                         Op *fromOp,
                                         Op *toOp,
                                         InIndex inIndex) const {

  Op::Settings settings(graph, "", fromOp->debugInfo.getId());

  auto ioCopyOp = std::make_unique<IoTileCopyOp>(
      Onnx::CustomOperators::IoTileCopy, settings);

  Op *ioCopy = ioCopyOp.get();
  graph.moveIntoGraph(std::move(ioCopyOp));

  // Copy the list of index's this input tensor is mapped
  auto indices = toOp->input->indices(tensor);

  // Remove this input tensor from the to op for each index
  logging::transform::debug(
      "Disconnecting in {} from {}:{}", tensor->id, toOp->debugName(), inIndex);
  toOp->disconnectInTensor(inIndex, tensor);

  ioCopy->connectInTensor(IoTileCopyOp::getInIndex(), tensor->id);

  TensorId copiedTensor = generateCopiedTensorId(tensor, toTileSet);

  ioCopy->createAndConnectOutTensor(0, copiedTensor);
  ioCopy->setup();

  // Add the copied input tensor to the to op for each index
  logging::transform::debug(
      "Connecting in {} to {}:{}", copiedTensor, toOp->debugName(), inIndex);
  toOp->connectInTensor(inIndex, copiedTensor);

  auto &sessionOptions = graph.getIr().getSessionOptions();
  bool isPhased =
      sessionOptions.virtualGraphMode == VirtualGraphMode::ExecutionPhases &&
      sessionOptions.executionPhaseSettings.phases;

  // If the schedule is interleaving instead of batched, we can schedule the
  // IoTileCopyOps as close to the consumers on the IO tiles as possible to
  // reduce the memory footprint on IO tiles.
  // Otherwise, follow a schedule that maximizes possible overlap
  bool tiedTopoCon =
      !isPhased || sessionOptions.executionPhaseSettings.schedule ==
                       ExecutionPhaseSchedule::Interleaving;

  if (fromTileSet == TileSet::IO) {
    // Copy direction: From IO tiles
    if (fromOp) {
      graph.topoCons->insert(fromOp, ioCopy, tiedTopoCon);
      ioCopy->settings = fromOp->settings;
    } else if (toOp) {
      ioCopy->settings = toOp->settings;
    }
    ioCopy->settings.name    = "";
    ioCopy->settings.tileSet = TileSet::Compute;
  }

  if (toTileSet == TileSet::IO) {
    // Copy direction: To IO tiles
    if (toOp) {
      graph.topoCons->insert(ioCopy, toOp, tiedTopoCon);
      ioCopy->settings = toOp->settings;
    } else if (fromOp) {
      ioCopy->settings = fromOp->settings;
    }
    ioCopy->settings.name    = "";
    ioCopy->settings.tileSet = TileSet::IO;
  }

  StreamingMemoryOpInserter::setPriority(
      ioCopy, isPhased, false, sessionOptions.executionPhaseSettings.schedule);
}

bool IoComputeTileCopy::apply(Graph &graph) const {
  // Keep a record of which tensors have been copied to/from IO tiles
  std::set<std::pair<TensorId, TileSet>> copiedTensors;

  auto schedule = graph.getOpSchedule({}, RequireOptimalSchedule::Yes);

  std::unordered_map<Op *, size_t> opScheduleIndex;
  for (size_t i = 0; i < schedule.size(); ++i) {
    opScheduleIndex.insert({schedule.at(i), i});
  }

  // For each op (in schedule order)
  for (Op *from : schedule) {

    TensorSet tensors;

    if (from->opid != Onnx::CustomOperators::IoTileCopy) {
      auto &input  = from->input;
      auto &output = from->output;

      // Any tensor without producer
      for (auto &t : input->tensorMap()) {
        Tensor *tensor = t.second;
        tensors.insert(tensor);
      }
      // Any tensor produced by this op
      for (auto &t : output->tensorMap()) {
        Tensor *tensor = t.second;
        tensors.insert(tensor);
      }

      // For each tensor
      for (auto *tensor : tensors) {

        // For each consumer op of the tensor
        // but, take a copy of the map as we will be modifying it.
        auto map = tensor->consumers.getMap();

        std::map<size_t, Op *> consumersInOrder;

        for (auto &kv : map) {
          auto it = opScheduleIndex.find(kv.first);
          if (it != opScheduleIndex.end()) {
            consumersInOrder.insert({it->second, kv.first});
          }
        }

        for (auto &c : consumersInOrder) {
          Op *to = c.second;

          if (to->opid != Onnx::CustomOperators::IoTileCopy) {

            for (auto inIndex : to->input->indices(tensor)) {
              auto fromTileSet =
                  tensor->getVirtualGraphIdAndTileSetUnsafe().second;
              auto toTileSet =
                  to->getIntrospectionInVirtualGraphId(inIndex).second;

              // If the ops have different IO tile status
              if (fromTileSet != toTileSet &&
                  fromTileSet != TileSet::Undefined &&
                  toTileSet != TileSet::Undefined) {

                logging::trace("[IoComputeTileCopy::apply] Copying {} from "
                               "tile set: {} to tile set: {}",
                               tensor->id,
                               fromTileSet,
                               toTileSet);

                bool alreadyCopied =
                    copiedTensors.find({tensor->id, toTileSet}) !=
                    copiedTensors.end();

                if (alreadyCopied == true) {
                  connectIoTileCopy(graph, tensor, toTileSet, to, inIndex);
                } else {
                  insertIoTileCopy(
                      graph, tensor, fromTileSet, toTileSet, from, to, inIndex);
                  // Record the copy
                  copiedTensors.insert({tensor->id, toTileSet});
                }
              }
            }
          }
        }
      }
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new IoComputeTileCopy);
}

} // namespace popart
