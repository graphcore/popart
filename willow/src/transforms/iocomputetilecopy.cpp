// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <popart/transforms/iocomputetilecopy.hpp>
#include <popart/transforms/streamingmemoryopinserter.hpp>

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
                                          Op *toOp) const {

  // Copy the list of index's this input tensor is mapped
  auto indices = toOp->input->indices(tensor);

  // Remove this input tensor from the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Disconnecting out {} from {}:{}", tensor->id, toOp->debugName(), i);
    toOp->disconnectInTensor(i, tensor);
  }

  TensorId copiedTensor =
      generateCopiedTensorId(tensor, toOp->settings.tileSet);

  // Add the copied input tensor to the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Connecting in {} from {}:{}", copiedTensor, toOp->debugName(), i);
    toOp->connectInTensor(i, copiedTensor);
  }
}

void IoComputeTileCopy::insertIoTileCopy(Graph &graph,
                                         Tensor *tensor,
                                         Op *fromOp,
                                         Op *toOp) const {

  Op::Settings settings(graph, "");

  auto ioCopyOp = std::make_unique<IoTileCopyOp>(
      Onnx::CustomOperators::IoTileCopy, settings);

  Op *ioCopy = ioCopyOp.get();
  graph.moveIntoGraph(std::move(ioCopyOp));

  // Copy the list of index's this input tensor is mapped
  auto indices = toOp->input->indices(tensor);

  // Remove this input tensor from the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Disconnecting in {} from {}:{}", tensor->id, toOp->debugName(), i);
    toOp->disconnectInTensor(i, tensor);
  }

  ioCopy->connectInTensor(IoTileCopyOp::getInIndex(), tensor->id);

  TensorId copiedTensor =
      generateCopiedTensorId(tensor, toOp->settings.tileSet);

  ioCopy->createAndConnectOutTensor(0, copiedTensor);
  ioCopy->setup();

  // Add the copied input tensor to the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Connecting in {} to {}:{}", copiedTensor, toOp->debugName(), i);
    toOp->connectInTensor(i, copiedTensor);
  }

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

  if (fromOp->settings.tileSet == TileSet::IO) {
    // Copy direction: From IO tiles
    graph.topoCons->insert(fromOp, ioCopy, tiedTopoCon);
    ioCopy->settings         = fromOp->settings;
    ioCopy->settings.name    = "";
    ioCopy->settings.tileSet = TileSet::Compute;
  }

  if (toOp->settings.tileSet == TileSet::IO) {
    // Copy direction: To IO tiles
    graph.topoCons->insert(ioCopy, toOp, tiedTopoCon);
    ioCopy->settings         = toOp->settings;
    ioCopy->settings.name    = "";
    ioCopy->settings.tileSet = TileSet::IO;
  }

  StreamingMemoryOpInserter::setPriority(
      ioCopy, isPhased, false, sessionOptions.executionPhaseSettings.schedule);
}

bool IoComputeTileCopy::apply(Graph &graph) const {
  // Keep a record of which tensors have been copied to/from IO tiles
  std::set<TensorId> copiedTensors;
  std::set<TensorId> processedTensors;

  auto schedule = graph.getOpSchedule({});

  std::unordered_map<Op *, size_t> opScheduleIndex;
  for (size_t i = 0; i < schedule.size(); ++i) {
    opScheduleIndex.insert({schedule.at(i), i});
  }

  // For each op (in schedule order)
  for (Op *from : schedule) {

    std::set<Tensor *> tensors;

    if (from->opid != Onnx::CustomOperators::IoTileCopy) {
      // For each input tensor
      auto &input  = from->input;
      auto &output = from->output;

      // Any tensor without producer: IO/Compute mapped to the first consumer
      for (auto &t : input->tensorMap()) {
        Tensor *tensor = t.second;
        if (tensor->tensorType() == TensorType::Stream ||
            tensor->tensorType() == TensorType::Const ||
            tensor->tensorType() == TensorType::Variable) {
          auto it = processedTensors.find(tensor->id);
          if (it == processedTensors.end()) {
            tensors.insert(tensor);
            processedTensors.insert(tensor->id);
          }
        }
      }
      // Any tensor produced by this op: IO/Compute mapped to this op
      for (auto &t : output->tensorMap()) {
        Tensor *tensor = t.second;
        auto it        = processedTensors.find(tensor->id);
        if (it == processedTensors.end()) {
          tensors.insert(tensor);
          processedTensors.insert(tensor->id);
        }
      }

      // For each tensor
      for (auto *tensor : tensors) {

        // For each consumer op of the tensor
        // but, take a copy of the map as we will be modifying it.
        auto &map = tensor->consumers.getMap();

        std::map<size_t, Op *> consumersInOrder;

        for (auto &kv : map) {
          consumersInOrder.insert({opScheduleIndex.at(kv.first), kv.first});
        }

        for (auto &c : consumersInOrder) {
          Op *to = c.second;

          if (to->opid != Onnx::CustomOperators::IoTileCopy) {

            // If the ops have different IO tile status
            if (from->settings.tileSet != to->settings.tileSet) {

              bool alreadyCopied =
                  copiedTensors.find(tensor->id) != copiedTensors.end();

              if (alreadyCopied == true) {
                connectIoTileCopy(graph, tensor, to);
              } else {
                insertIoTileCopy(graph, tensor, from, to);
                // Record the copy
                copiedTensors.insert(tensor->id);
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
