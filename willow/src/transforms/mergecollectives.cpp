// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popart/transforms/mergecollectives.hpp>

namespace popart {
std::size_t MergeCollectivesTransform::id() {
  return typeid(MergeCollectivesTransform).hash_code();
}

bool MergeCollectivesTransform::apply(Graph &graph) const {
  std::set<OpId> includeOps;
  for (OpId id : graph.getOpIds()) {
    includeOps.insert(id);
  }
  applyToOps(graph, includeOps);
  return true;
}

std::vector<Op *>
MergeCollectivesTransform::applyToOps(Graph &graph,
                                      const std::set<OpId> includeOps) const {
  std::vector<Op *> createdOps;
  std::vector<Op *> opSchedule =
      graph.getOpSchedule({}, RequireOptimalSchedule::Yes);
  std::vector<std::string> scheduleNames;
  for (Op *op : opSchedule) {
    scheduleNames.emplace_back("\t" + op->debugName() + "\n");
  }
  logging::debug("Schedule for merging: \n {}", scheduleNames);

  // Touch each op in the schedule exactly once
  for (std::vector<Op *>::iterator schedulePos = opSchedule.begin();
       schedulePos != opSchedule.end();
       schedulePos++) {
    Op *op = *schedulePos;
    if (includeOps.count(op->id) > 0) {
      if (graph.getIr()
              .getSessionOptions()
              .replicatedCollectivesSettings.mergeAllReduceCollectives) {
        if (auto candidate = dynamic_cast<ReplicatedAllReduceOp *>(op)) {
          Op *new_op = attemptToMergeOnOp<MultiReplicatedAllReduceOp>(
              candidate, schedulePos, opSchedule);
          if (new_op) {
            createdOps.emplace_back(new_op);
          }
        }
      }
      // TODO T56323 support merging reducescatter collectives
      // TODO T56324 support merging allgather collectives
    }
  }

  return createdOps;
}

template <typename MultiOpType, typename BaseType>
std::unique_ptr<MultiOpType> MergeCollectivesTransform::constructMultiOp(
    BaseType *baseOp,
    std::vector<bool> modifiesIndexInplace,
    std::vector<TensorInfo> outInfoFromBaseOps,
    std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
    std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet) const {

  CommGroup group                   = baseOp->getGCLCommGroup();
  CollectiveOperator collectiveType = baseOp->getCollectiveOp();

  // Construct the multi collective operation
  auto multiOp = std::make_unique<MultiOpType>(
      collectiveType,
      group,
      popart::Op::Settings(baseOp->getGraph(), "MultiCollectiveOp"),
      modifiesIndexInplace,
      outInfoFromBaseOps,
      inputVirtualGraphIdAndTileSet,
      outputVirtualGraphIdAndTileSet);
  multiOp->setup();

  multiOp->setVirtualGraphId(baseOp->getVirtualGraphId());
  multiOp->settings.executionContext = baseOp->settings.executionContext;
  if (baseOp->hasPipelineStage()) {
    multiOp->setPipelineStage(baseOp->getPipelineStage());
  }
  if (baseOp->hasExecutionPhase()) {
    multiOp->setExecutionPhase(baseOp->getExecutionPhase());
  }
  return multiOp;
}

template <typename MultiOpType, typename BaseType>
Op *MergeCollectivesTransform::attemptToMergeOnOp(
    BaseType *baseOp,
    std::vector<Op *>::iterator &schedulePos,
    std::vector<Op *> &opSchedule) const {
  auto &graph = baseOp->getGraph();

  // Ops which can be merged with the current baseOp must
  // have been constrained to form a contiguous segment in the schedule
  // using topocons in the ContiguateCollectivesForMerging transform
  // The collectives must also match in the other properties
  auto requiredDataType =
      baseOp->inTensor(baseOp->getInIndex())->info.data_type();
  auto requiredGCLGroup          = baseOp->getGCLCommGroup();
  auto requiredCollectiveOp      = baseOp->getCollectiveOp();
  auto &requiredExecutionContext = baseOp->settings.executionContext;

  // Keep iterating through the schedule until the next
  // non-matching op is reached
  std::vector<BaseType *> matchingOps;
  while (schedulePos != opSchedule.end()) {
    Op *op = *schedulePos;
    if (BaseType *candidate = dynamic_cast<BaseType *>(op)) {
      bool dtypeCheck =
          candidate->inTensor(candidate->getInIndex())->info.data_type() ==
          requiredDataType;
      bool groupCheck = candidate->getGCLCommGroup() == requiredGCLGroup;
      bool collectiveCheck =
          candidate->getCollectiveOp() == requiredCollectiveOp;
      bool executionContextCheck =
          candidate->settings.executionContext == requiredExecutionContext;

      // An op must pass all the checks to be a match
      if (dtypeCheck && groupCheck && collectiveCheck &&
          executionContextCheck) {
        matchingOps.emplace_back(candidate);
        ++schedulePos;
        if (schedulePos != opSchedule.end()) {
          // Make sure the current op and the next op are tied together
          // in the schedule
          Op *nextOp               = *schedulePos;
          std::vector<Op *> afters = graph.topoCons->getTiedAfters(op);
          if (std::find(afters.begin(), afters.end(), nextOp) != afters.end()) {
            continue; // continue onto the next op in the schedule
          }
        }
      }
    }
    // Break out of the loop
    // - if next op is not a matching collective operation
    // - or, the next op is not tied to the previous one
    // - or, reached the end of schedule
    break;
  }

  // Notify which ops are about to me merged
  std::vector<std::string> allMatchNames;
  for (auto op : matchingOps) {
    allMatchNames.emplace_back("\t" + op->debugName() + "\n");
  }
  logging::info("[MergeCollectivesTransform ] Merging op {} with its "
                "contiguous matches: \n{}",
                baseOp->debugName(),
                allMatchNames);

  // The matchingOps can now be connected to the same op
  // collect all:
  // 1. outInfo
  // 2. inputs, outputs and their vgraph and tilesets
  // 3. linked tensor ids and their vgraph and tilesets
  std::vector<TensorInfo> outInfoFromBaseOps;
  for (auto op : matchingOps) {
    outInfoFromBaseOps.emplace_back(op->outInfo(op->getOutIndex()));
  }

  // Determine which parts of the multiop can be inplaced. Even if the original
  // base op is inplace, in the multiOp there can be aliases between
  // the inputs, which prevents full inplacing.
  AliasModel popMem;
  AliasModelGrower aliasModelGrower{popMem};
  aliasModelGrower.growFullGraph(graph, DataDependenciesOnly::No);
  std::vector<bool> modifiesIndexInplace;
  for (size_t i = 0; i < matchingOps.size(); i++) {
    auto opI      = matchingOps.at(i);
    bool modifies = opI->modifiesIndex(opI->getInIndex());
    modifiesIndexInplace.emplace_back(modifies);
    popart::Tensor *inputI = opI->inTensor(opI->getInIndex());
    auto aliases           = popMem.allAliases(*inputI);
    for (size_t j = 0; j < i; j++) {
      auto opJ               = matchingOps.at(j);
      popart::Tensor *inputJ = opJ->inTensor(opJ->getInIndex());
      if (std::find(aliases.begin(), aliases.end(), inputJ) != aliases.end()) {
        modifiesIndexInplace[i] = false;
        modifiesIndexInplace[j] = false;
        logging::warn("[MergeCollectivesTransform::attemptToMergeOnOp] "
                      " Inputs {} and {} are aliases. Collective operation "
                      " on these tensors will be outplaced.",
                      inputI->str(),
                      inputJ->str());
      }
    }
  }

  std::vector<popart::Tensor *> inputTensors;
  std::vector<popart::Tensor *> outputTensors;
  std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet;
  std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet;
  for (auto op : matchingOps) {
    auto input   = op->inTensor(op->getInIndex());
    auto output  = op->outTensor(op->getOutIndex());
    auto visited = std::set<OpId>{op->id};
    inputTensors.emplace_back(input);
    inputVirtualGraphIdAndTileSet.emplace_back(
        input->getVirtualGraphIdAndTileSet(visited));
    outputVirtualGraphIdAndTileSet.emplace_back(
        output->getVirtualGraphIdAndTileSet(visited));
    outputTensors.emplace_back(output);
  }

  std::vector<popart::Tensor *> linkedIndexTensors;
  if (baseOp->hasInput(baseOp->getCollectiveLinkedIndex())) {
    for (auto op : matchingOps) {
      auto visited           = std::set<OpId>{op->id};
      auto linkedIndexTensor = op->inTensor(op->getCollectiveLinkedIndex());
      linkedIndexTensors.emplace_back(linkedIndexTensor);
      inputVirtualGraphIdAndTileSet.emplace_back(
          linkedIndexTensor->getVirtualGraphIdAndTileSet(visited));
    }
  }

  for (auto op : matchingOps) {
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
  }

  // Construct new op
  std::unique_ptr<MultiOpType> multiOp =
      constructMultiOp<MultiOpType>(baseOp,
                                    modifiesIndexInplace,
                                    outInfoFromBaseOps,
                                    inputVirtualGraphIdAndTileSet,
                                    outputVirtualGraphIdAndTileSet);
  OpId multiOpId = multiOp->id;

  // Connect input, outputs and linked indices
  InIndex inputIndex{0};
  for (auto inputTensor : inputTensors) {
    multiOp->connectInTensor(inputIndex, inputTensor->id);
    inputIndex++;
  }
  for (auto inputTensor : linkedIndexTensors) {
    multiOp->connectInTensor(inputIndex, inputTensor->id);
    inputIndex++;
  }

  OutIndex outIndex{0};
  for (auto outputTensor : outputTensors) {
    multiOp->connectOutTensor(outIndex, outputTensor->id);
    outIndex++;
  }

  // Move into graph
  logging::info("[MergeCollectivesTransform] Combined op is {}",
                multiOp->debugName());
  graph.moveIntoGraph(std::move(multiOp));

  // Cleanup the old ops
  for (auto op : matchingOps) {
    graph.eraseOp(op->id);
  }
  return graph.getOp(multiOpId);
}

namespace {
bool init = Transform::registerTransform(new MergeCollectivesTransform);
}

} // namespace popart