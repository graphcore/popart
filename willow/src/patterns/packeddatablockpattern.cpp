// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/init.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/packeddatablock.hpp>
#include <popart/op/printtensor.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/sequenceslice.hpp>
#include <popart/patterns/packeddatablockpattern.hpp>
#include <popart/util.hpp>

namespace popart {

namespace {

// Try set an ops virtual graph id using the input and output tensors.
void trySetVGraphId(Op *op) {
  // Try set the vgraph using inputs.
  for (auto t : op->input->tensors()) {
    if (t->hasVirtualGraphId()) {
      op->setVirtualGraphId(t->getVirtualGraphId());
      return;
    }
  }

  // Try set vgraph using outputs.
  for (auto t : op->output->tensors()) {
    if (t->hasVirtualGraphId()) {
      op->setVirtualGraphId(t->getVirtualGraphId());
      return;
    }
  }
}

// Try to set the virtual graph id for ops in a graph.
// Used on the loop body graph to set the id of the additional ops.
void updateVGraphInformation(Graph &graph) {
  // Collect all ops with no virtual graph id.
  std::vector<Op *> noVGraphOps;
  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (!op->hasVirtualGraphId()) {
      noVGraphOps.push_back(op);
    }
  }

  while (!noVGraphOps.empty()) {
    for (auto op : noVGraphOps) {
      trySetVGraphId(op);
    }

    // Remove from noVGraphOps, ops whose virtual graph id was set.
    // Error if no ops were removed, as that will result in an infinite loop.
    auto opCount = noVGraphOps.size();
    noVGraphOps.erase(
        std::remove_if(noVGraphOps.begin(), noVGraphOps.end(), [](auto op) {
          return op->hasVirtualGraphId();
        }));
    if (opCount == noVGraphOps.size()) {
      // I don't think we're likely to hit this error. For it to occur, there
      // needs to be an op/ops with no virtual graph id, that are unconnected
      // from the ops with virtual graph ids.
      // An infinite loop is a pain to debug though, so I'm including this check
      // anyway.
      throw internal_error("No ops remove, possible infinite loop.");
    }
  }
}

// This will call `Ir::createIntermediateTensorId`, but first check to see if
// `tid` ends with the postfix `__t\d+`, and removes it if present.
TensorId createIntermediateTensorId(const TensorId &tid, Ir &ir) {
  auto isNum = [](char c) { return c >= '0' and c <= '9'; };

  int64_t i = tid.size() - 1;
  if (isNum(tid.at(i))) {
    while (isNum(tid.at(i))) {
      i -= 1;
    }
    if (tid.at(i) == 't' and tid.at(i - 1) == '_' and tid.at(i - 2) == '_') {
      return ir.createIntermediateTensorId(tid.substr(0, i - 2));
    }
  }

  return ir.createIntermediateTensorId(tid);
}

// Helper method to add a SequenceSliceOp to a graph.
TensorId addSequenceSliceOp(TensorId source,
                            TensorId dest,
                            TensorId N,
                            TensorId sourceOffsets,
                            TensorId destOffsets,
                            bool zeroUnused,
                            Graph &graph,
                            Op::Settings settings) {
  auto ss = graph.createOp<SequenceSliceOp>(
      Onnx::CustomOperators::SequenceSlice_1, zeroUnused, settings);
  ss->connectInTensor(0, source);
  ss->connectInTensor(1, dest);
  ss->connectInTensor(2, N);
  ss->connectInTensor(3, sourceOffsets);
  ss->connectInTensor(4, destOffsets);
  ss->createAndConnectOutTensor(
      0, createIntermediateTensorId(source, graph.getIr()));
  ss->setup();
  return ss->outId(0);
}

// Helper method to add an InitOp to a graph.
TensorId addInitOp(TensorInfo resultInfo,
                   InitType initType,
                   Graph &graph,
                   Op::Settings settings,
                   TensorId name) {
  auto initOp = graph.createOp<InitOp>(Onnx::CustomOperators::Init_1,
                                       resultInfo,
                                       TensorType::ActGrad,
                                       initType,
                                       settings);
  initOp->createAndConnectOutTensor(
      InitOp::getOutIndex(), createIntermediateTensorId(name, graph.getIr()));
  initOp->setup();
  return initOp->outId(InitOp::getOutIndex());
}

// Helper method to add a ReshapeOp to a graph.
TensorId addReshapeOp(const TensorId &t,
                      Shape newShape,
                      Graph &graph,
                      Op::Settings settings) {
  auto reshape = graph.createOp<ReshapeOp>(
      Onnx::AiOnnx::OpSet9::Reshape, newShape, settings);
  reshape->connectInTensor(ReshapeOp::getInIndex(), t);
  reshape->createAndConnectOutTensor(
      ReshapeOp::getOutIndex(), createIntermediateTensorId(t, graph.getIr()));
  reshape->setup();
  return reshape->outId(ReshapeOp::getOutIndex());
}

// Add a ReshapeOp that reshapes `t` to a single dimensional tensor.
TensorId addFlattenOp(const TensorId &t, Graph &graph, Op::Settings settings) {
  return addReshapeOp(
      t, {graph.getTensors().get(t)->info.nelms()}, graph, settings);
}

// Helper method to add a DynamicSliceOp to a graph.
TensorId addDynamicSliceOp(const TensorId &inId,
                           const TensorId &indexId,
                           const std::vector<int64_t> &axes,
                           const std::vector<int64_t> &sizes,
                           Graph &graph,
                           Op::Settings settings) {
  auto dso = graph.createOp<DynamicSliceOp>(
      Onnx::CustomOperators::DynamicSlice_1, axes, sizes, true, settings);
  dso->connectInTensor(DynamicSliceOp::getInIndex(), inId);
  dso->connectInTensor(DynamicSliceOp::getIndexInIndex(), indexId);

  dso->createAndConnectOutTensor(
      DynamicSliceOp::getOutIndex(),
      createIntermediateTensorId(inId, graph.getIr()));
  dso->setup();
  return dso->outId(DynamicSliceOp::getOutIndex());
}

// Take a slice of size `sliceSize` from a tensor using a tensor index.
// Equivalent to the python expression `result = tid[index:index+sliceSize]`.
TensorId indexTensor(TensorId tid,
                     TensorId index,
                     int64_t sliceSize,
                     Graph &graph,
                     Op::Settings settings) {
  auto tinfo = graph.getTensors().get(tid)->info;
  tid =
      addReshapeOp(tid, {tinfo.dim(0) / sliceSize, sliceSize}, graph, settings);
  tid = addDynamicSliceOp(tid, index, {0}, {1}, graph, settings);
  tid = addFlattenOp(tid, graph, settings);
  return tid;
}

// Create a constant tensor with uniformly spaced offset indices, used for
// unpacking sequences.
// Equivalent to the python:
//   `[i * maxTokensPerSequence for i in range(callbackBatchSize)]`
// For the inner loop, where `callbackBatchSize` sequences are unpacked
// into a tensor of size [callbackBatchSize * maxTokensPerSequence, ...].
TensorId getInnerOffsets(Graph &graph,
                         int64_t callbackBatchSize,
                         int64_t maxTokensPerSequence) {
  std::vector<uint32_t> tInnerOffsetsData;
  for (int i = 0; i < callbackBatchSize; i++) {
    tInnerOffsetsData.push_back(i * maxTokensPerSequence);
  }

  TensorId tInnerOffsets = addScope(
      graph, graph.getIr().createIntermediateTensorId("tInnerOffsets"));
  graph.getTensors().addConstInit(tInnerOffsets,
                                  {DataType::UINT32, {callbackBatchSize}},
                                  tInnerOffsetsData.data());
  return tInnerOffsets;
}

struct CallbackIO {
  TensorId loopIteration;
  TensorId loopCondition;
  std::vector<PackedSequences> sequenceInputs;
  PackedSequences result;
};

// Add inputs to the callbackGraph for that are required for use with the
// LoopOp.
CallbackIO setupCallbackInputs(PackedDataBlockOp *op, const TensorId resultId) {
  // The callbackGraph should already have an input for each data input.
  // For the loop op, we need to add inputs for:
  //   loopIndex, loopCondition, result, and each length and offset tensor.
  CallbackIO inputInfo;

  auto &graph = op->getCalledGraph();
  // Get the current state of the inputs before changing them.
  std::vector<TensorId> dataInputs = graph.getInputIds();

  // For the loop op, the first two graph inputs are always the loop index and
  // condition.
  inputInfo.loopIteration = addScope(graph, "loopIndex");
  inputInfo.loopCondition = addScope(graph, "loopCondition");
  graph.addInput(LoopOp::getLoopIterationInIndex(),
                 inputInfo.loopIteration,
                 {DataType::INT32, {}},
                 false);
  graph.addInput(LoopOp::getTerminationConditionInIndex(),
                 inputInfo.loopCondition,
                 {DataType::BOOL, {}},
                 false);

  // The result needs to be a persistent input, and these need to be the first
  // inputs.
  auto &resultInfo    = op->outInfo(0);
  auto scopedResultId = addScope(graph, resultId);
  graph.addInput(
      LoopOp::getFirstInputInIndex(), scopedResultId, resultInfo, false);

  // Inputs should only be added once. The inputs and result may share lengths
  // and offsets.
  auto tryAddCallbackInput = [&](Tensor *t) {
    auto scopedId = addScope(graph, t->id);
    if (!graph.hasInputId(scopedId)) {
      graph.addInput(scopedId, t->info);
    }
    return graph.getTensors().get(scopedId);
  };

  // Add the offset and length inputs for the inputs
  auto packedInputs = op->getPackedInputs();
  for (int i = 0; i < packedInputs.size(); i++) {
    auto &packedInput = packedInputs.at(i);
    auto offsets      = tryAddCallbackInput(packedInput.offsets);
    auto lengths      = tryAddCallbackInput(packedInput.lengths);
    auto data         = graph.getTensors().get(dataInputs.at(i));
    inputInfo.sequenceInputs.push_back(PackedSequences{data, offsets, lengths});
  }

  // Add the offset and length input for the result
  auto packedResult  = op->getPackedOutput();
  auto resultOffsets = tryAddCallbackInput(packedResult.offsets);
  auto resultLengths = tryAddCallbackInput(packedResult.lengths);
  auto resultData    = graph.getTensors().get(scopedResultId);
  inputInfo.result = PackedSequences{resultData, resultOffsets, resultLengths};

  return inputInfo;
}

// For each `PackedSequences` in `inputInfo`, add the ops to index the `offsets`
// and `lengths` tensors, and replace `offsets` and `lengths` in the
// `PackedSequence` with the indexed versions.
// After this `offsets` and `lengths` will be of shape [callbackBatchSize].
void indexOffsetsAndLengths(CallbackIO &inputInfo,
                            PackedDataBlockOp *op,
                            Op::Settings settings) {
  std::map<Tensor *, Tensor *> alreadyIndexed;
  auto callbackBatchSize = op->getCallbackBatchSize();

  auto &graph = op->getCalledGraph();

  auto getIndexed = [&](Tensor *t) {
    auto found = alreadyIndexed.find(t);
    if (found != alreadyIndexed.end()) {
      return found->second;
    } else {
      auto indexedId = indexTensor(
          t->id, inputInfo.loopIteration, callbackBatchSize, graph, settings);
      auto indexedTensor = graph.getTensors().get(indexedId);

      alreadyIndexed.insert({t, indexedTensor});
      return indexedTensor;
    }
  };

  // Index the inputs
  for (auto &ps : inputInfo.sequenceInputs) {
    ps.offsets = getIndexed(ps.offsets);
    ps.lengths = getIndexed(ps.lengths);
  }

  // Index the results
  inputInfo.result.offsets = getIndexed(inputInfo.result.offsets);
  inputInfo.result.lengths = getIndexed(inputInfo.result.lengths);
}

// Create the tensors into which the packed sequence inputs will be unpacked.
std::vector<TensorId> createDestinationTensors(CallbackIO &inputInfo,
                                               PackedDataBlockOp *pdb,
                                               Op::Settings settings) {
  std::vector<TensorId> result;

  auto destInfos = pdb->callbackSequenceInInfos();
  auto &graph    = pdb->getCalledGraph();

  for (int i = 0; i < inputInfo.sequenceInputs.size(); i++) {
    auto &packedInput = inputInfo.sequenceInputs.at(i);

    auto destInfo = destInfos.at(i);

    auto dest = addInitOp(
        destInfo, InitType::NoInit, graph, settings, packedInput.data->id);
    result.push_back(dest);
  }

  return result;
}

// Add SequenceSliceOps to slice the callbackGraph input tensors, and then
// reconnect the consumers of those tensors to use the sliced tensor.
void sequenceSliceCallbackInputs(
    CallbackIO &inputInfo,
    PackedDataBlockOp *pdb,
    const std::vector<TensorId> &destinationTensors,
    Op::Settings settings) {
  int64_t callbackBatchSize = pdb->getCallbackBatchSize();
  auto maxSequenceLengths   = pdb->getMaxSequenceLengths();
  auto &graph               = pdb->getCalledGraph();

  // Add the sequence slice ops
  for (int i = 0; i < inputInfo.sequenceInputs.size(); i++) {
    auto &packedInput = inputInfo.sequenceInputs.at(i);
    auto consumers    = packedInput.data->consumers.getOps();
    auto inShape      = packedInput.data->info.shape();
    auto dest         = destinationTensors.at(i);

    packedInput.data->info = pdb->inInfo(pdb->dataIndex(i));

    auto innerOffsets =
        getInnerOffsets(graph, callbackBatchSize, maxSequenceLengths.at(i));

    auto out = addSequenceSliceOp(packedInput.data->id,
                                  dest,
                                  packedInput.lengths->id,
                                  packedInput.offsets->id,
                                  innerOffsets,
                                  true,
                                  graph,
                                  settings);

    out = addReshapeOp(out, inShape, graph, settings);

    // Swap all ops using data to use dest
    for (auto op : consumers) {
      auto indices = op->input->indicesMap().at(packedInput.data);
      for (auto idx : indices) {
        op->disconnectInTensor(idx);
        op->connectInTensor(idx, out);
      }
    }
  }
}

// Add a SequenceSliceOp to insert the result of each loop iteration into the
// result destination.
void sequenceSliceCallbackOutput(CallbackIO &inputInfo,
                                 PackedDataBlockOp *pdb,
                                 Op::Settings &settings) {
  auto &graph     = pdb->getCalledGraph();
  auto graphOutId = graph.getOutputId(1);
  auto graphOut   = graph.getTensors().get(graphOutId);

  // Get an output shape and remove the batch dimension
  auto graphOutShape  = graphOut->info.shape();
  graphOutShape.at(1) = graphOutShape.at(1) * graphOutShape.at(0);
  graphOutShape.erase(graphOutShape.begin());

  graphOutId = addReshapeOp(graphOutId, graphOutShape, graph, settings);

  auto innerOffsets =
      getInnerOffsets(graph,
                      pdb->getCallbackBatchSize(),
                      graphOutShape.at(0) / pdb->getCallbackBatchSize());

  auto out = addSequenceSliceOp(graphOutId,
                                inputInfo.result.data->id,
                                inputInfo.result.lengths->id,
                                innerOffsets,
                                inputInfo.result.offsets->id,
                                false,
                                graph,
                                settings);

  // Change the outputs on the graph
  graph.markAsOutput(1, out, true);
}

// Create the LoopOp that will replace the PackedDataBlockOp.
void createLoopOp(PackedDataBlockOp *pdb, const TensorId &resultId) {
  auto &graph    = pdb->getGraph();
  auto &loopBody = pdb->getCalledGraph();
  auto loopOp    = dynamic_cast<LoopOp *>(graph.createOp<LoopOp>(
      Onnx::AiOnnx::OpSet11::Loop, Op::Settings(graph, ""), loopBody));
  pdb->transferBaseProperties(loopOp);

  loopOp->setTripCountValue(pdb->getCallbackIterations());
  loopOp->connectInTensor(2, resultId);

  // Connect the PackedDataBlockOp inputs to the loopOp.
  std::set<TensorId> addedInputs;
  for (auto &idx_tensor : pdb->input->tensorMap()) {
    auto idx    = idx_tensor.first;
    auto tensor = idx_tensor.second;

    // Only add the inputs once.
    if (addedInputs.find(tensor->id) == addedInputs.end()) {
      loopOp->connectInTensor(idx + 3, tensor->id);
      addedInputs.insert(tensor->id);
    }
  }

  // Connect the PackedDataBlockOp outputs to the loopOp.
  auto outputs = pdb->output->tensorMap();
  for (auto &idx_tensor : outputs) {
    auto idx    = idx_tensor.first;
    auto tensor = idx_tensor.second;
    pdb->disconnectOutTensor(tensor);
    loopOp->connectOutTensor(idx, tensor->id);
  }

  loopOp->setup();
}

} // namespace

bool PackedDataBlockPattern::matches(Op *op) const {
  return op->isConvertibleTo<PackedDataBlockOp>();
}

std::vector<const Tensor *> PackedDataBlockPattern::touches(Op *op) const {
  return {};
}

bool PackedDataBlockPattern::apply(Op *op) const {
  // We know that this cast should work from the check in `matches`.
  auto packedDataBlock = dynamic_cast<PackedDataBlockOp *>(op);
  auto &graph          = op->getGraph();
  auto &callbackGraph  = packedDataBlock->getCalledGraph();

  auto resultInfo    = packedDataBlock->outInfo(0);
  auto resultOuterId = addInitOp(
      resultInfo, InitType::Zero, graph, packedDataBlock->settings, "result");

  auto inputInfo = setupCallbackInputs(packedDataBlock, resultOuterId);

  // Add the output for the loopCondition
  callbackGraph.markAsOutput(0, inputInfo.loopCondition, false);

  Op::Settings callbackSettings{callbackGraph, ""};

  indexOffsetsAndLengths(inputInfo, packedDataBlock, callbackSettings);

  auto destTensors =
      createDestinationTensors(inputInfo, packedDataBlock, callbackSettings);

  sequenceSliceCallbackInputs(
      inputInfo, packedDataBlock, destTensors, callbackSettings);

  sequenceSliceCallbackOutput(inputInfo, packedDataBlock, callbackSettings);

  createLoopOp(packedDataBlock, resultOuterId);

  if (graph.getIr().getSessionOptions().virtualGraphMode !=
      VirtualGraphMode::Off) {
    updateVGraphInformation(packedDataBlock->getCalledGraph());
  }

  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  graph.eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<PackedDataBlockPattern>
    PackedDataBlockPattern("PackedDataBlock");
}

} // namespace popart
